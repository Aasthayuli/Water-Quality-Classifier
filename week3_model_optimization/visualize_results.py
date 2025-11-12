import sys, os
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from week1_data_preprocessing.data_preprocessing import test_dataset
from week2_Model_Training.water_cnn_model import WaterCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, "week2_Model_Training", "water_quality_classifier.pth")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(OUTPUTS_DIR, "graphs")
PRED_DIR = os.path.join(OUTPUTS_DIR, "predictions")
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# Load model
model = WaterCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Visualization of sample predictions
for i in range(10):
    image, true_label = test_dataset[i]
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        _, predicted = torch.max(output, 1)
        label = predicted.item()
        image = image * 0.5 + 0.5  # Unnormalize

        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"Predicted: {label}, Actual: {true_label}")
        plt.axis('off')
        plt.savefig(os.path.join(PRED_DIR, f"prediction_{i+1}.png"))
        plt.close()

print(f"Prediction images saved to: {PRED_DIR}")

# Example static graph (accuracy summary)
plt.figure(figsize=(5, 4))
plt.bar(["Accuracy"], [90], color="green")  # you can pass actual accuracy if needed
plt.ylim(0, 100)
plt.ylabel("Percentage")
plt.title("Model Test Accuracy (example)")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(os.path.join(GRAPHS_DIR, f"accuracy_plot_{timestamp}.png"))
plt.close()

print(f"Graphs saved to: {GRAPHS_DIR}")
