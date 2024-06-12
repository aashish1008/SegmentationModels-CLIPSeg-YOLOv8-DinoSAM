# Import necessary libraries and modules
from transformers import CLIPSegProcessor, CLIPSegVisionModel, CLIPSegForImageSegmentation
from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt

# Load the pre-trained CLIPSeg processor and model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# Define the URL of the image to be processed
url = "https://variety.com/wp-content/uploads/2020/04/tiger-king-netflix.jpg"

# Open the image from the URL
image = Image.open(requests.get(url, stream=True).raw)

# Define the text prompts for segmentation
texts = ["a person", "a tiger"]

# Process the image and text inputs for the model
inputs = processor(text=texts, images=[image] * len(texts), padding=True, return_tensors="pt")

# Run the model inference without calculating gradients
with torch.no_grad():
    outputs = model(**inputs)

# Extract the logits (raw model outputs) from the inference results
logits = outputs.logits

# Print the shape of the logits tensor
print(logits.shape)

# Add an extra dimension to the logits tensor for visualization purposes
logits = logits.unsqueeze(1)

# Set up the matplotlib plot with subplots for each text prompt
_, ax = plt.subplots(1, len(texts) + 1, figsize=(3 * (len(texts) + 1), 12))

# Remove axis ticks for each subplot
[a.axis('off') for a in ax.flatten()]

# Display the original image in the first subplot
ax[0].imshow(image)

# Display the segmentation masks for each text prompt in the subsequent subplots
[ax[i + 1].imshow(torch.sigmoid(logits[i][0])) for i in range(len(texts))]

# Add text labels to the subplots indicating the prompts
[ax[i + 1].text(0, -15, prompt) for i, prompt in enumerate(texts)]

# Show the plot with the original image and segmentation masks
plt.show()
