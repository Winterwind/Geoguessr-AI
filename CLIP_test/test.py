from PIL import Image
import torch, clip, os, base64, folium
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from io import BytesIO
from tqdm import tqdm
from datasets import load_dataset
from pprint import pp
import prototype

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
    model = prototype.GeoGuessr().to(my_device)
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    checkpoint = torch.load('output/checkpoints/best_model.pt', map_location=my_device)
    num_epochs = 10
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    sample_test = dataset['train'].select(range(10000, len(dataset['train'])))
    test_data = prototype.GeoGuessrDataset(sample_test, processor)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss: {checkpoint['val_loss']:.2f} km")

    # Test phase
    os.makedirs('output/test_predictions', exist_ok=True)

    test_loss = 0
    image_counter = 0
    max_images_to_save = 20  # Limit number of saved images
    model.eval()

    print("\nStarting test phase...")
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=True)
        for batch_idx, (pixel_values, true_coords) in enumerate(test_pbar):
            pixel_values = pixel_values.to(my_device)
            true_coords = true_coords.to(my_device)

            if use_amp:
                with autocast(device_type='cuda'):
                    predicted_coords = model(pixel_values)
            else:
                predicted_coords = model(pixel_values)

            # Calculate loss for the batch (mean)
            loss = prototype.haversine_loss(predicted_coords, true_coords)
            test_loss += loss.detach().cpu().numpy()
            
            avg_test_loss = test_loss / (batch_idx + 1)
            test_pbar.set_postfix({'loss': f'{loss.item():.2f} km', 'avg_loss': f'{avg_test_loss:.2f} km'})
            
            # Calculate individual distances for visualization
            individual_distances = prototype.haversine_distance(predicted_coords, true_coords)

            # Save visualizations for first few batches
            if image_counter < max_images_to_save:
                # Move data to CPU for processing
                pred_cpu = predicted_coords.cpu()
                true_cpu = true_coords.cpu()
                images_cpu = pixel_values.cpu()
                distances_cpu = individual_distances.cpu()
                
                # Process each image in the batch
                for i in range(len(pixel_values)):
                    if image_counter >= max_images_to_save:
                        break
                    
                    # Get individual predictions and ground truth
                    pred_lat, pred_lon = pred_cpu[i][0].item(), pred_cpu[i][1].item()
                    true_lat, true_lon = true_cpu[i][0].item(), true_cpu[i][1].item()
                    
                    # Get the pre-calculated distance for this image
                    distance_km = distances_cpu[i].item()
                    
                    # Calculate percent error
                    max_distance = 20000
                    percent_error = (distance_km / max_distance) * 100
                    
                    # Denormalize image
                    image = images_cpu[i]
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                    image = image * std + mean
                    image = torch.clamp(image, 0, 1)
                    
                    to_pil = ToPILImage()
                    pil_image = to_pil(image)
                    
                    # ========== CREATE MAP WITH FOLIUM ==========
                    # Calculate center point between predicted and true locations
                    center_lat = (pred_lat + true_lat) / 2
                    center_lon = (pred_lon + true_lon) / 2
                    
                    # Create folium map
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=4,  # Adjust based on distance
                        tiles='OpenStreetMap'
                    )
                    
                    # Add marker for true location (green)
                    folium.Marker(
                        location=[true_lat, true_lon],
                        popup=f'True Location<br>({true_lat:.4f}°, {true_lon:.4f}°)',
                        tooltip='True Location',
                        icon=folium.Icon(color='green', icon='info-sign')
                    ).add_to(m)
                    
                    # Add marker for predicted location (red)
                    folium.Marker(
                        location=[pred_lat, pred_lon],
                        popup=f'Predicted Location<br>({pred_lat:.4f}°, {pred_lon:.4f}°)',
                        tooltip='Predicted Location',
                        icon=folium.Icon(color='red', icon='info-sign')
                    ).add_to(m)
                    
                    # Draw line between them
                    folium.PolyLine(
                        locations=[[true_lat, true_lon], [pred_lat, pred_lon]],
                        color='blue',
                        weight=2,
                        opacity=0.7,
                        popup=f'Distance: {distance_km:.2f} km'
                    ).add_to(m)
                    
                    # Adjust zoom to show both points
                    m.fit_bounds([[true_lat, true_lon], [pred_lat, pred_lon]])
                    
                    # Save map as HTML
                    map_path = f'output/test_predictions/test_map_{image_counter:03d}.html'
                    m.save(map_path)
                    
                    # ========== CREATE FIGURE WITH IMAGE ==========
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.imshow(pil_image)
                    ax.axis('off')
                    
                    # Create title
                    title = f"Test Image {image_counter + 1}\n"
                    title += f"Predicted: ({pred_lat:.4f}°, {pred_lon:.4f}°)\n"
                    title += f"True: ({true_lat:.4f}°, {true_lon:.4f}°)\n"
                    title += f"Distance Error: {distance_km:.2f} km\n"
                    title += f"Percent Error: {percent_error:.2f}%"
                    
                    plt.title(title, fontsize=12, pad=20)
                    plt.tight_layout()
                    
                    # Save the figure
                    save_path = f'output/test_predictions/test_img_{image_counter:03d}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved {save_path} and {map_path} - Distance: {distance_km:.2f} km")
                    
                    image_counter += 1

    avg_test_loss = test_loss / len(test_loader)

    print(f"\nTest Phase Complete. Average test loss: {avg_test_loss:.2f} km")
    print(f"Saved {image_counter} test prediction visualizations to output/test_predictions/")

    # Plot loss
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='train')
    plt.plot(epochs, val_losses, 'orange', label='valid')
    plt.axhline(y=avg_test_loss, color='r', linestyle='-', label='test')
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss (km)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/loss.png')
    plt.close()