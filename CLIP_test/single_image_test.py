from PIL import Image
import torch, folium
import torch.nn as nn
from transformers import CLIPModel
import matplotlib.pyplot as plt
from transformers import CLIPProcessor
from torchvision.transforms import ToPILImage
import os

class GeoGuessr(nn.Module):
    def __init__(self, clip_model_name="geolocal/StreetCLIP", unfreeze_layers=2):
        super().__init__()
        # Load pre-trained CLIP as feature extractor
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP weights 
        for param in self.clip.parameters():
            param.requires_grad = False

        # Unfreeze last N layers of CLIP
        if unfreeze_layers > 0:
            total_layers = len(self.clip.vision_model.encoder.layers)
            print(f"Total CLIP vision encoder layers: {total_layers}")
            print(f"Unfreezing last {unfreeze_layers} layers")
        
        for i in range(total_layers - unfreeze_layers, total_layers):
            for param in self.clip.vision_model.encoder.layers[i].parameters():
                param.requires_grad = True
        
        # Unfreeze the final layer norm and projection
        for param in self.clip.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        if hasattr(self.clip.vision_model, 'visual_projection'):
            for param in self.clip.vision_model.visual_projection.parameters():
                param.requires_grad = True
        
        # Get the embedding dimension from CLIP
        embed_dim = self.clip.config.projection_dim # 768
        
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
        image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
        
        # Predict coordinates
        coords = self.regressor(image_embeds)
        
        # Constrain to valid lat/lon ranges
        lat = torch.tanh(coords[:, 0]) * 90  # [-90, 90]
        lon = torch.tanh(coords[:, 1]) * 180  # [-180, 180]
        
        return torch.stack([lat, lon], dim=1)
    
if __name__ == '__main__':
    use_amp = False
    if torch.backends.mps.is_available():
        print('Using mps')
        my_device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using cuda')
        my_device = torch.device("cuda")
        use_amp = True
    else:
        print('Using cpu')
        my_device = torch.device("cpu")

    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler('cuda')
        print("Using mixed precision training (AMP)")
    else:
        scaler = None
        print("Using standard precision training")
    
    model = GeoGuessr().to(my_device)
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    checkpoint = torch.load('output_best_FT/checkpoints/best_model.pt', map_location=my_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss: {checkpoint['val_loss']:.2f} km")

    image_path = "single_test.png"
    img = Image.open(image_path).convert('RGB')
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(my_device)

    predicted_coords = model(pixel_values)
    pred_cpu = predicted_coords.cpu()
    images_cpu = pixel_values.cpu()

    # Get individual predictions and ground truth
    pred_lat, pred_lon = pred_cpu[0][0].item(), pred_cpu[0][1].item()

    # Denormalize image
    image = images_cpu[0]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
                    
    to_pil = ToPILImage()
    pil_image = to_pil(image)
                    
    # ========== CREATE MAP WITH FOLIUM ==========
    # Calculate center point between predicted and true locations
    center_lat = (pred_lat)
    center_lon = (pred_lon)
                    
    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,  # Adjust based on distance
        tiles='OpenStreetMap'
    )
                    
    # Add marker for predicted location (red)
    folium.Marker(
        location=[pred_lat, pred_lon],
        popup=f'Predicted Location<br>({pred_lat:.4f}°, {pred_lon:.4f}°)',
        tooltip='Predicted Location',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
                    
    # Adjust zoom to show both points
    # m.fit_bounds([[pred_lat, pred_lon]])

    os.makedirs('single_guess', exist_ok=True)
                    
    # Save map as HTML
    map_path = f'single_guess/single_guess.html'
    m.save(map_path)
                    
    # ========== CREATE FIGURE WITH IMAGE ==========
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(pil_image)
    ax.axis('off')
                    
    # Create title
    title = f"Single Guess\n"
    title += f"Predicted: ({pred_lat:.4f}°, {pred_lon:.4f}°)\n"
                    
    plt.title(title, fontsize=12, pad=20)
    plt.tight_layout()
                    
    # Save the figure
    save_path = f'single_guess/single_guess.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
                    
    print(f"Saved {save_path} and {map_path}")