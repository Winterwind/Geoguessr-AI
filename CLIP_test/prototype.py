from PIL import Image
import torch, clip
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from pprint import pp