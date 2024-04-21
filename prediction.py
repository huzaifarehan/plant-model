import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image
import os

# Model definition
def get_model(num_classes):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=7).to(device)  # Assuming the number of classes is 7
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the image to predict

# image_paths = ['./plant1.jpg','./plant2.jpg','./plant3(2).jpg' ,'./plant3.jpg','./plant4.jpg','./plant5.jpg','./plant6.jpg','./plant7.jpg','./plant-test.jpg']
image_paths=['./plant3_test.jpg','./plant3_test(1).jpg','plant4_test.jpg','./plant5_test.jpg']
for image_path in image_paths:
    image = Image.open(image_path)

    # Apply the transformation to the image
    input_image = transform(image).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(input_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get predicted class
    predicted_class = torch.argmax(probabilities).item()
    class_name = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7'][predicted_class]

    print(f"Predicted class: {class_name}, Probability: {probabilities[predicted_class].item()}")
