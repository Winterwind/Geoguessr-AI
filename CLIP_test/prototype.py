from PIL import Image
import torch, clip
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from pprint import pp

ds = load_dataset("stochastic/random_streetview_images_pano_v0.0.2")

if torch.backends.mps.is_available():
    print('use mps')
    my_device = torch.device("mps")
elif torch.cuda.is_available():
    print('use cuda')
    my_device = torch.device("cuda")
else:
    print('use cpu')
    my_device = torch.device("cpu")

class GeoGuessr(nn.Module):
    def __init__(self, clip_model_name="geolocal/StreetCLIP"):
        super().__init__()
        # Load pre-trained CLIP as feature extractor
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP weights (optional - can fine-tune later)
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # Get the embedding dimension from CLIP
        embed_dim = self.clip.config.projection_dim  # Usually 512
        
        # Regression head for coordinates
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: [latitude, longitude]
        )
        
    def forward(self, pixel_values):
        # Get image embeddings from CLIP
        with torch.no_grad():  # Remove if fine-tuning CLIP
            image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
        
        # Predict coordinates
        coords = self.regressor(image_embeds)
        
        # Constrain to valid lat/lon ranges
        lat = torch.tanh(coords[:, 0]) * 90  # [-90, 90]
        lon = torch.tanh(coords[:, 1]) * 180  # [-180, 180]
        
        return torch.stack([lat, lon], dim=1)

# Training setup
model = GeoGuessr()
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

# Custom loss that accounts for Earth's spherical geometry
def haversine_loss(pred, target):
    """
    Calculate great circle distance between predicted and actual coordinates
    More accurate than euclidean distance for lat/lon
    """
    lat1, lon1 = pred[:, 0], pred[:, 1]
    lat2, lon2 = target[:, 0], target[:, 1]
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(a))
    
    # Earth radius in km
    r = 6371
    
    return (c * r).mean()

# Training loop example
optimizer = torch.optim.Adam(model.regressor.parameters(), lr=1e-4)

def train_step(images, true_coords):
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs['pixel_values'].to(my_device)
    
    predicted_coords = model(pixel_values)
    loss = haversine_loss(predicted_coords, true_coords)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()