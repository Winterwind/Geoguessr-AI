from PIL import Image
import torch, clip
from transformers import CLIPProcessor, CLIPModel
from pprint import pp

model = CLIPModel.from_pretrained("geolocal/StreetCLIP")

#Uncomment the 2 next lines to run the model with GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
if torch.backends.mps.is_available():
    print('use mps')
    my_device = torch.device("mps")
elif torch.cuda.is_available():
    print('use cuda')
    my_device = torch.device("cuda")
else:
    print('use cpu')
    my_device = torch.device("cpu")

processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

#Define the list of countries
labels = ['Albania', 'Andorra', 'Argentina', 'Australia', 'Austria', 'Bangladesh', 'Belgium', 'Bermuda', 'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Bulgaria', 'Cambodia', 'Canada', 'Chile', 'China', 'Colombia', 'Croatia', 'Czech Republic', 'Denmark', 'Dominican Republic', 'Ecuador', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Greenland', 'Guam', 'Guatemala', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan', 'Kenya', 'Kyrgyzstan', 'Laos', 'Latvia', 'Lesotho', 'Lithuania', 'Luxembourg', 'Macedonia', 'Madagascar', 'Malaysia', 'Malta', 'Mexico', 'Monaco', 'Mongolia', 'Montenegro', 'Netherlands', 'New Zealand', 'Nigeria', 'Norway', 'Pakistan', 'Palestine', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Romania', 'Russia', 'Rwanda', 'Senegal', 'Serbia', 'Singapore', 'Slovakia', 'Slovenia', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Swaziland', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 'United States', 'Uruguay']
# text = clip.tokenize(labels).to(my_device)
# print(f'Available models: {clip.available_models()}')


def classify(image):
    #Uncomment the next line if you want to compute score with GPU and comment the one right after
    #inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # image_features = outputs.encode_image(image)
        # text_features = outputs.encode_text(text)

        logits_per_image = outputs.logits_per_image
        prediction = logits_per_image.softmax(dim=1)
    #Compute classification score for each country
    confidences = {labels[i]: float(prediction[0][i].item()) for i in range(len(labels))}
    return confidences

#Input image
image_path = 'turkey.jpg'
img = Image.open(image_path)

#Compute classification score
scores = classify(img)
#Sort the result and take the top 10
sorted_countries = sorted(scores.items(), key=lambda x:x[1], reverse=True)
sorted_scores = dict(sorted_countries)

import itertools
top10 = dict(itertools.islice(sorted_scores.items(), 10))

#print the result
pp(top10)
# print(f'Image features: {image_features}')