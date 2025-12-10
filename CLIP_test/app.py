from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import torch
import base64
import io
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# Load model
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")

class GeoGuessr(nn.Module):
    def __init__(self, clip_model_name="geolocal/StreetCLIP", unfreeze_layers=2):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        for param in self.clip.parameters():
            param.requires_grad = False

        if unfreeze_layers > 0:
            total_layers = len(self.clip.vision_model.encoder.layers)
            for i in range(total_layers - unfreeze_layers, total_layers):
                for param in self.clip.vision_model.encoder.layers[i].parameters():
                    param.requires_grad = True
        
        for param in self.clip.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        if hasattr(self.clip.vision_model, 'visual_projection'):
            for param in self.clip.vision_model.visual_projection.parameters():
                param.requires_grad = True
        
        embed_dim = self.clip.config.projection_dim
        
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        
    def forward(self, pixel_values):
        image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
        coords = self.regressor(image_embeds)
        lat = torch.tanh(coords[:, 0]) * 90
        lon = torch.tanh(coords[:, 1]) * 180
        return torch.stack([lat, lon], dim=1)

model = GeoGuessr(unfreeze_layers=2).to(device)
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

# Load checkpoint - UPDATE THIS PATH
checkpoint = torch.load('output/checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded on {device}")
print(f"Best model from epoch {checkpoint['epoch']+1}, val loss: {checkpoint['val_loss']:.2f} km")

def base64_to_pil(base64_string):
    """Convert base64 string to PIL Image"""
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

@app.route('/')
def index():
    """Serve the HTML interface"""
    return send_file('geoguessr_ui.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict coordinates from image"""
    try:
        data = request.json
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Convert base64 to PIL Image
        image = base64_to_pil(data['image'])
        
        # Process image
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            # Get prediction
            coords = model(pixel_values)
            
            lat = coords[0][0].item()
            lon = coords[0][1].item()
        
        print(f"Predicted: ({lat:.4f}, {lon:.4f})")
        
        return jsonify({
            'latitude': lat,
            'longitude': lon
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'device': str(device)})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("GeoGuessr AI Server")
    print("="*50)
    print(f"Device: {device}")
    print(f"Navigate to: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)