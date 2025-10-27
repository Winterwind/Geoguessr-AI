from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

# Load model
model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

# Load and process image
image = Image.open('turkey.jpg')
inputs = processor(images=image, return_tensors="pt")

# Extract features
with torch.no_grad():
    image_embeds = model.get_image_features(pixel_values=inputs['pixel_values'])

print(f"Shape of embeddings: {image_embeds.shape}")  # torch.Size([1, 512])
print(f"First 10 values: {image_embeds[0, :10]}")
# Example output: tensor([-0.2341,  0.5123, -0.1234,  0.8901, ...])