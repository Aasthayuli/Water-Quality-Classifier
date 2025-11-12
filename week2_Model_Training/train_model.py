import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from week1_data_preprocessing.data_preprocessing import train_loader, val_loader
from week2_Model_Training.water_cnn_model import WaterCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "water_quality_classifier.pth")

model = WaterCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

best_val_loss = float('inf')
patience, trigger_times = 5, 0
num_epochs = 25

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
    scheduler.step(avg_val_loss)

    # Early Stopping + Saving
    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model improved. Saved at epoch {epoch+1}.")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch+1}.")
            break

print("Model training completed and saved at:", MODEL_PATH)
