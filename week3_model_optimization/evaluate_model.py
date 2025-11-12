import sys, os
import torch
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from week1_data_preprocessing.data_preprocessing import test_loader
from week2_Model_Training.water_cnn_model import WaterCNN

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, "week2_Model_Training", "water_quality_classifier.pth")

OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Load model
model = WaterCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Evaluate on test data
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Metrics
acc = 100 * np.mean(np.array(y_true) == np.array(y_pred))
print(f"Test Accuracy: {acc:.2f}%")

# Classification report and confusion matrix
report = classification_report(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(LOGS_DIR, f"evaluation_log_{timestamp}.txt")

with open(log_file, "w") as f:
    f.write(f"Test Accuracy: {acc:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    np.savetxt(f, conf_matrix, fmt="%d")

print(f"Evaluation log saved to: {log_file}")
