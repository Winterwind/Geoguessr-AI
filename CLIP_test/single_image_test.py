from PIL import Image
import torch, folium, prototype
import matplotlib.pyplot as plt
from transformers import CLIPProcessor
from torchvision.transforms import ToPILImage
import os

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

    dataset = prototype.ds
    model = prototype.GeoGuessr(unfreeze_layers=2).to(my_device)
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    checkpoint = torch.load('output_best_FT/checkpoints/best_model.pt', map_location=my_device)

    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss: {checkpoint['val_loss']:.2f} km")

    # sample = dataset['train'][0]
    # image = sample['image']
    # print(type(image))

    image_path = "single_test.png"
    img = Image.open(image_path)
    # img = img.resize((3030, 561))
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