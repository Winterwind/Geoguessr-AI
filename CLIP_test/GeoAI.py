import torch, os
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from datasets import load_dataset
from tqdm import tqdm
import random

class GeoGuessrDataset(Dataset):
    def __init__(self, hf_dataset, processor):
        self.dataset = hf_dataset
        self.processor = processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]  # This returns a dict
        
        image = sample['image']
        lat = float(sample['latitude'])
        lon = float(sample['longitude'])
        # print(f'{image}\n{lat}\n{lon}')
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dim
        
        # Create coordinate tensor
        coords = torch.tensor([lat, lon], dtype=torch.float32)
        
        return pixel_values, coords

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

# Custom loss that accounts for Earth's spherical geometry
def haversine_distance(pred, target, reduction=None):
    """
    Calculate great circle distance between predicted and actual coordinates
    
    Args:
        pred: predicted coordinates [batch_size, 2]
        target: true coordinates [batch_size, 2]
        reduction: 'mean' for average, 'none' for individual distances
    
    Returns:
        distance in km (scalar if reduction='mean', tensor if reduction='none')
    """
    lat1, lon1 = pred[:, 0], pred[:, 1]
    lat2, lon2 = target[:, 0], target[:, 1]
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    a = torch.clamp(a, 0.0, 1.0)
    c = 2 * torch.asin(torch.sqrt(a))
    
    # Earth radius in km
    r = 6371
    distances = c * r
    
    if reduction == 'mean':
        return distances.mean()
    elif reduction is None:
        return distances
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def haversine_loss(pred, target):
    return haversine_distance(pred, target, reduction='mean')

seed = random.randint(0, 10000)
seed = 5519 # Fixed seed for reproducibility, comment out to use random seed

os.makedirs('output', exist_ok=True)
with open('output/log.txt', 'w') as f:
    f.write(f"Seed: {seed}\n")

ds = load_dataset("stochastic/random_streetview_images_pano_v0.0.2").shuffle(seed=seed)
print(f"Using seed: {seed}")

# Initialize processor
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

if __name__ == '__main__':
    if torch.backends.mps.is_available():
        print('Using mps')
        my_device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using cuda')
        my_device = torch.device("cuda")
    else:
        print('Using cpu')
        my_device = torch.device("cpu")

    # Prepare datasets
    sample_train = ds['train'].select(range(9000))
    sample_val = ds['train'].select(range(9000, 10000))
    sample_test = ds['train'].select(range(10000, len(ds['train'])))

    train_data = GeoGuessrDataset(sample_train, processor)
    test_data = GeoGuessrDataset(sample_test, processor)
    val_data = GeoGuessrDataset(sample_val, processor)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # Initialize model
    model = GeoGuessr(unfreeze_layers=2).to(my_device)

    # Verify what's trainable
    print("\nTrainable parameters:")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(f"  {name}: {param.numel():,} params")

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100*trainable_params/total_params:.2f}%\n")

    # Initialize GradScaler for mixed precision training (only for CUDA)
    use_amp = torch.cuda.is_available()
    if use_amp:
        from torch.amp import autocast, GradScaler
        scaler = GradScaler('cuda')
        print("Using mixed precision training (AMP)")
    else:
        scaler = None
        print("Using standard precision training")

    clip_params = []
    for name, param in model.clip.named_parameters():
        if param.requires_grad:
            clip_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': model.regressor.parameters(), 'lr': 1e-4},  # Higher LR for regressor
        {'params': clip_params, 'lr': 1e-5}  # Lower LR for CLIP (10x smaller)
    ])

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # minimize validation loss
        factor=0.5,           # reduce LR by half
        patience=3,           # wait 3 epochs before reducing
        min_lr=1e-6,
        threshold=10,
        threshold_mode='abs'  # use absolute threshold        
    )

    os.makedirs('output/checkpoints', exist_ok=True)
    print(f"Using {len(sample_train)} training images, {len(sample_val)} validation images, and {len(sample_test)} testing images\n")

    # RESUME TRAINING
    resume_from_checkpoint = False # Set to True to resume
    checkpoint_path = 'output/checkpoints/best_model.pt'

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    start_epoch = 0
    num_epochs = 30
    patience = 2
    epochs_no_improve = 0
    best_val_loss = float('inf')

    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=my_device)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Clean up
        del checkpoint
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load on CPU to avoid GPU memory issues
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load training history
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)

        # Clean up
        del checkpoint
        gc.collect()
        
        print(f"Resumed from epoch {start_epoch}")
        print(f"Previous best val loss: {best_val_loss:.2f}")
        print(f"Epochs without improvement: {epochs_no_improve}")

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            if i == 0:
                print(f"Regressor LR: {lr:.2e}")
            elif i == 1:
                print(f"CLIP LR: {lr:.2e}")

        print(f"Scheduler num bad epochs: {scheduler.num_bad_epochs}")
        print(f"Starting from epoch: {start_epoch + 1}\n")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=True)
        for batch_idx, (pixel_values, true_coords) in enumerate(train_pbar):
            pixel_values = torch.FloatTensor(pixel_values).to(my_device)
            true_coords = true_coords.to(my_device)
            
            optimizer.zero_grad()
            
            # Use autocast only if AMP is enabled (CUDA only)
            if use_amp:
                with autocast(device_type='cuda'):
                    predicted_coords = model(pixel_values)
                    loss = haversine_loss(predicted_coords, true_coords)
                
                # Scale loss and do backward pass
                scaler.scale(loss).backward()

                # Clip gradients to prevent explosion
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for MPS or CPU
                predicted_coords = model(pixel_values)
                loss = haversine_loss(predicted_coords, true_coords)
                
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_train_loss += loss.item()
            avg_train_loss = total_train_loss / (batch_idx + 1)

            train_pbar.set_postfix({'loss': f'{loss.item():.2f} km', 'avg_loss': f'{avg_train_loss:.2f} km'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.2f}")

        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", leave=True)
            for batch_idx, (pixel_values, true_coords) in enumerate(val_pbar):
                pixel_values = pixel_values.to(my_device)
                true_coords = true_coords.to(my_device)

                if use_amp:
                    with autocast(device_type='cuda'):
                        predicted_coords = model(pixel_values)
                else:
                    predicted_coords = model(pixel_values)

                loss = haversine_loss(predicted_coords, true_coords)
                total_val_loss += loss.item()
                avg_val_loss = total_val_loss / (batch_idx + 1)

                val_pbar.set_postfix({'loss': f'{loss.item():.2f} km', 'avg_loss': f'{avg_val_loss:.2f} km'})
                    
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.2f}\n")

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_no_improve': epochs_no_improve
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Save the best model (lowest validation loss)
            best_checkpoint_path = 'output/checkpoints/best_model.pt'
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Saved best model: {best_checkpoint_path} (Val Loss: {avg_val_loss:.2f})")
        else:
            epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered.")
                break

    # Load best model for testing
    print("\nLoading best model for testing...")
    best_checkpoint = torch.load('output/checkpoints/best_model.pt', map_location=my_device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {best_checkpoint['epoch']+1} with validation loss: {best_checkpoint['val_loss']:.2f}")

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
            loss = haversine_loss(predicted_coords, true_coords)
            test_loss += loss.detach().cpu().numpy()
            avg_test_loss = test_loss / (batch_idx + 1)

            test_pbar.set_postfix({'loss': f'{loss.item():.2f} km', 'avg_loss': f'{avg_test_loss:.2f} km'})
            
            # Calculate individual distances for visualization
            individual_distances = haversine_distance(predicted_coords, true_coords)

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
                    
                    # Calculate percent error (relative to Earth's half-circumference ~20,000 km)
                    max_distance = 20000  # km
                    percent_error = (distance_km / max_distance) * 100
                    
                    # Convert tensor image to displayable format
                    # Denormalize the image (CLIP uses mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                    image = images_cpu[i]
                    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                    image = image * std + mean
                    image = torch.clamp(image, 0, 1)
                    
                    # Convert to PIL Image
                    to_pil = ToPILImage()
                    pil_image = to_pil(image)
                    
                    # Create figure with image and information
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
                    
                    print(f"Saved {save_path} - Distance: {distance_km:.2f} km")
                    
                    image_counter += 1

        avg_test_loss = test_loss / len(test_loader)
        avg_test_score = -avg_test_loss

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

    with open('output/log.txt', 'a') as f:
        f.write(f"Best validation loss: {best_val_loss:.2f} km\n")
        f.write(f"Best training loss: {min(train_losses):.2f} km\n")
        f.write(f"Average test loss: {avg_test_loss:.2f} km\n")