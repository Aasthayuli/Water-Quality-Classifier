import os
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

for main_folder in ['train', 'test']:
    for folder in ['clean', 'muddy', 'polluted']:
        path = f'data/water_dataset/{main_folder}/{folder}'
        for file in os.listdir(path):
            fpath = os.path.join(path, file)
            try:
                Image.open(fpath).verify()
            except:
                print("Corrupted removed:", fpath)
                os.remove(fpath)

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(128, scale=(0.7, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder(root='data/water_dataset/train', transform=train_transform)
test_dataset = datasets.ImageFolder(root='data/water_dataset/test', transform=test_transform)

val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Preprocessing completed successfully.")
print("Classes:", train_dataset.classes)