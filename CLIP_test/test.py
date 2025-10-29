from PIL import Image
import torch, clip, tqdm, os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from datasets import load_dataset
from pprint import pp

checkpoint = torch.load('output/checkpoints/best_model.pt')
num_epochs = 10
train_losses = checkpoint['train_losses']
val_losses = [value * 10 for value in checkpoint['val_losses']]
test_losses = [2575.63] * num_epochs

# Plot loss
plt.figure(figsize=(10, 6))
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, 'b-', label='train')
plt.plot(epochs, val_losses, 'orange', label='valid')
plt.plot(epochs, test_losses, 'r-', label='test')
plt.title('Geoguessr Model Loss')
plt.xlabel('epoch')
plt.ylabel('loss (km)')
plt.legend()
plt.grid(True)
plt.savefig('output/loss.png')
plt.close()