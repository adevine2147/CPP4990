import torch
from torchvision import transforms
from PIL import Image
from cnn_template import SimpleCNN

model_path = ''

model = SimpleCNN()
model.load_state_dict(torch.load(model_path))
model.eval()
image_path = ''
image_size = 28

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Preprocess the image
image = Image.open(image_path).convert('RGB')  # If your image is RGB, convert to grayscale
image = transform(image)
image = image.unsqueeze(0)  # Add a batch dimension

# Inference
with torch.no_grad():
    print("entered torch nograd")
    model.eval()

    # Forward pass
    output = model(image)

# Convert output to probabilities (optional) or get predicted class
probabilities = torch.nn.functional.softmax(output[0], dim=0)
predicted_class = torch.argmax(probabilities).item()

# Display or use the results
print(f'Predicted Class: {predicted_class}')
print(f'Class Probabilities: {probabilities.numpy()}')
