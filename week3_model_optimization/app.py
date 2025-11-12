import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
from pathlib import Path
import requests

# Model Definition

class WaterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Model Loading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WaterCNN().to(device)

# Path of model file (in same folder as this script)
MODEL_PATH = Path(__file__).resolve().parent / "water_quality_classifier.pth"

# Download model if not found
if not MODEL_PATH.exists():
    st.info("Downloading model, please wait...")
    url = "https://drive.google.com/uc?export=download&id=1pXSDJB2EfMUPfJeeNkQcAKc2YEE3jDeh"
    response = requests.get(url)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("Model downloaded successfully!")
    else:
        st.error("Model download failed. Please check the link.")

# Load model
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    st.error("Model file not found — please ensure it exists in the same folder.")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# Streamlit App UI

st.title("💧 Water Quality Classifier")

uploaded_file = st.file_uploader("Upload an image of water", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = predicted.item()

    classes = ['clean', 'muddy', 'polluted']
    st.success(f"Prediction: **{classes[label].capitalize()} Water** 💧")
