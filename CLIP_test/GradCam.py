import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
import cv2
from GeoAI import GeoGuessr, GeoGuessrDataset, processor, haversine_distance
import os

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: Your GeoGuessr model
            target_layer: The layer to visualize (e.g., model.clip.vision_model.encoder.layers[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        # For transformer layers, output is typically a tuple
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()
    
    def generate_cam(self, input_tensor, target_coord_idx=0):
        """
        Generate CAM for a specific output (latitude=0 or longitude=1)
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            target_coord_idx: 0 for latitude, 1 for longitude
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass on the target output
        target = output[0, target_coord_idx]
        target.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [seq_len, hidden_dim]
        activations = self.activations[0]  # [seq_len, hidden_dim]
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=0)  # [hidden_dim]
        
        # Weighted combination of activation maps
        cam = (weights.unsqueeze(0) * activations).sum(dim=1)  # [seq_len]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize_cam(self, image, cam, alpha=0.5):
        """
        Overlay CAM on the original image
        
        Args:
            image: PIL Image or numpy array [H, W, 3]
            cam: CAM values [seq_len] - needs reshaping to spatial dimensions
            alpha: Overlay transparency
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        
        # seq_len = 577 (24*24) + 1 (CLS token)
        # Remove CLS token if present
        if len(cam) == 577:
            cam = cam[1:]  # Remove CLS token
        
        # Reshape CAM to spatial dimensions
        grid_size = int(np.sqrt(len(cam)))
        cam_reshaped = cam.reshape(grid_size, grid_size)
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam_reshaped, (w, h))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = (heatmap * alpha + image * (1 - alpha)).astype(np.uint8)
        
        return overlay

# Example usage
if __name__ == '__main__':
    seed = 5519 # Should match seed used to train model
    ds = load_dataset("stochastic/random_streetview_images_pano_v0.0.2").shuffle(seed=seed)
    print(f"Using seed: {seed}")

    if torch.backends.mps.is_available():
        print('Using mps')
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print('Using cuda')
        device = torch.device("cuda")
    else:
        print('Using cpu')
        device = torch.device("cpu")

    # Load your trained model
    model = GeoGuessr(unfreeze_layers=2).to(device)
    checkpoint = torch.load('output/checkpoints/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Choose target layer (last unfrozen layer)
    target_layer = model.clip.vision_model.encoder.layers[-1]
    
    # Create GradCAM
    gradcam = GradCAM(model, target_layer)
    
    # Get test images
    max_images = 20
    sample_test = ds['train'].select(range(10000, 10000 + max_images))
    test_data = GeoGuessrDataset(sample_test, processor)
    
    # Create output directory
    os.makedirs('output_best_FT/gradcam_predictions', exist_ok=True)
    
    print(f"Generating GradCAM visualizations for {max_images} test images...\n")
    
    for img_idx in range(max_images):
        # Process one image
        pixel_values, true_coords = test_data[img_idx]
        pixel_values_batch = pixel_values.unsqueeze(0).to(device)
        true_coords_tensor = true_coords.unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            predicted_coords = model(pixel_values_batch)
        
        # Calculate distance
        distance_km = haversine_distance(predicted_coords, true_coords_tensor).item()
        
        # Calculate percent error
        max_distance = 20000  # km
        percent_error = (distance_km / max_distance) * 100
        
        # Get coordinates
        pred_lat, pred_lon = predicted_coords[0][0].item(), predicted_coords[0][1].item()
        true_lat, true_lon = true_coords[0].item(), true_coords[1].item()
        
        # Generate CAMs for latitude and longitude
        cam_lat = gradcam.generate_cam(pixel_values_batch, target_coord_idx=0)
        cam_lon = gradcam.generate_cam(pixel_values_batch, target_coord_idx=1)
        
        # Denormalize the tensor image
        pixel_np = pixel_values.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        pixel_np = (pixel_np * std + mean) * 255
        pixel_np = np.clip(pixel_np, 0, 255).astype(np.uint8)
        
        # Visualize
        overlay_lat = gradcam.visualize_cam(pixel_np, cam_lat)
        overlay_lon = gradcam.visualize_cam(pixel_np, cam_lon)
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(pixel_np)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(overlay_lat)
        axes[1].set_title('GradCAM - Latitude', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(overlay_lon)
        axes[2].set_title('GradCAM - Longitude', fontsize=12)
        axes[2].axis('off')
        
        # Add overall title with prediction info
        title = f"Test Image {img_idx + 1}\n"
        title += f"Predicted: ({pred_lat:.4f}°, {pred_lon:.4f}°)\n"
        title += f"True: ({true_lat:.4f}°, {true_lon:.4f}°)\n"
        title += f"Distance Error: {distance_km:.2f} km | Percent Error: {percent_error:.2f}%"
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Save
        save_path = f'output_best_FT/gradcam_predictions/gradcam_{img_idx:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {save_path} - Distance: {distance_km:.2f} km, Error: {percent_error:.2f}%")
    
    print(f"\nGenerated {max_images} GradCAM visualizations in output/gradcam_predictions/")