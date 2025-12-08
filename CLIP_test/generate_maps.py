from PIL import Image
import torch, os, folium
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from datasets import load_dataset
from tqdm import tqdm
import random

pred_lat = 9.6388
pred_lon = 40.6097
true_lat = -35.8930
true_lon = 137.3106

def haversine_distance(lat1, lon1, lat2, lon2, reduction=None):
    """
    Calculate great circle distance between predicted and actual coordinates
    
    Args:
        pred: predicted coordinates [batch_size, 2]
        target: true coordinates [batch_size, 2]
        reduction: 'mean' for average, 'none' for individual distances
    
    Returns:
        distance in km (scalar if reduction='mean', tensor if reduction='none')
    """
    # lat1, lon1 = pred[:, 0], pred[:, 1]
    # lat2, lon2 = target[:, 0], target[:, 1]
    lat1 = torch.Tensor([lat1])
    lat2 = torch.Tensor([lat2])
    lon1 = torch.Tensor([lon1])
    lon2 = torch.Tensor([lon2])
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
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
    
distance_km = haversine_distance(pred_lat, pred_lon, true_lat, true_lon, reduction='mean')

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
map_path = f'output/test_sample_right.html'
m.save(map_path)