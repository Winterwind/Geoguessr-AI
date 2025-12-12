import torch, os
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from GeoAI_points import GeoGuessr, GeoGuessrDataset, processor, haversine_distance, geoguessr_score

seed = 5519 # Should match seed used to train model
ds = load_dataset("stochastic/random_streetview_images_pano_v0.0.2").shuffle(seed=seed)
print(f"Using seed: {seed}")

if torch.backends.mps.is_available():
    print('Using mps')
    my_device = torch.device("mps")
elif torch.cuda.is_available():
    print('Using cuda')
    my_device = torch.device("cuda")
else:
    print('Using cpu')
    my_device = torch.device("cpu")

model = GeoGuessr().to(my_device)

use_amp = torch.cuda.is_available()
if use_amp:
    from torch.amp import autocast, GradScaler
    scaler = GradScaler('cuda')
    print("Using mixed precision training (AMP)")
else:
    scaler = None
    print("Using standard precision training")

test_set = ds['train'].select(range(10000, len(ds['train'])))
test_data = GeoGuessrDataset(test_set, processor)
test_loader = DataLoader(test_data, batch_size=20, shuffle=False)

os.makedirs('tests', exist_ok=True)

best_checkpoint = torch.load('output/checkpoints/best_model.pt', map_location=my_device)
model.load_state_dict(best_checkpoint['model_state_dict'])
train_losses = best_checkpoint['train_losses']
val_losses = best_checkpoint['val_losses']

test_loss = 0
distances = []
scores = []
model.eval()

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

        # Calculate individual distances and scores
        individual_distances = haversine_distance(predicted_coords, true_coords, reduction=None)
        individual_scores = geoguessr_score(predicted_coords, true_coords)
        
        # Store distances and scores for std calculation
        distances.extend(individual_distances.cpu().numpy())
        scores.extend(individual_scores.cpu().numpy())
        
        # Calculate batch mean for progress tracking
        loss = individual_distances.mean()
        test_loss += loss.item()
        
        avg_test_loss = test_loss / (batch_idx + 1)
        avg_test_score = np.mean(scores)
        test_pbar.set_postfix({'loss': f'{loss.item():.2f} km', 'avg_loss': f'{avg_test_loss:.2f} km'})

avg_test_loss = test_loss / len(test_loader)
std_test_loss = np.std(distances)

avg_test_score = np.mean(scores)
std_test_score = np.std(scores)

print(f"\nTest Phase Complete.")
print(f"\n=== Distance Metrics ===")
print(f"Average distance: {avg_test_loss:.2f} km")
print(f"Std deviation: {std_test_loss:.2f} km")
print(f"Min distance: {np.min(distances):.2f} km")
print(f"Max distance: {np.max(distances):.2f} km")
print(f"Median distance: {np.median(distances):.2f} km")

print(f"\n=== GeoGuessr Score Metrics ===")
print(f"Average score: {avg_test_score:.2f} / 5000")
print(f"Std deviation: {std_test_score:.2f}")
print(f"Min score: {np.min(scores):.2f}")
print(f"Max score: {np.max(scores):.2f}")
print(f"Median score: {np.median(scores):.2f}")