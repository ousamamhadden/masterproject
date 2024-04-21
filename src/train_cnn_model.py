import numpy as np
import pandas as pd
import os
import util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime

CSV_FILE = 'data/raw/airbnb-listings.csv'
IMAGE_FOLDER = 'data/raw/images'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#torch.set_num_threads()


####################
# Data preparation #
####################
df = util.load_data_airbnb(CSV_FILE, pbar_size=494954)

indices_to_remove = []
existing_images = set(os.listdir(IMAGE_FOLDER))

# Remove rows with no corresponding image from df
print("Removing rows without images...")
print("Shape before:", df.shape)
for index, row in df.iterrows():
    image_filename = f"{row['ID']}_thumb.jpg"
    
    if image_filename not in existing_images:
        indices_to_remove.append(index)

df = df.drop(indices_to_remove)
print("Shape after:", df.shape)

# Remove rows which are too recent
print("Removing rows with first review after 2017...")
print("Shape before:", df.shape)
df.dropna(subset=['First Review'], inplace=True)
df.dropna(subset=['Reviews per Month'], inplace=True)
df = df[df['First Review'] <= '2016-12-31']
print("Shape after:", df.shape)


# Assign labels
labels = [0, 1, 2] # Low, medium, high
df['Reviews per Month (Class)'] = pd.qcut(df["Reviews per Month"],
                                           q=len(labels),
                                           labels=labels)
print(df['Reviews per Month (Class)'].value_counts())

grouped = df.groupby('Reviews per Month (Class)')
for label, group in grouped:
    min_value = group['Reviews per Month'].min()
    max_value = group['Reviews per Month'].max()
    print(f"For {label}: Min Value: {min_value}, Max Value: {max_value}")

filter_size = 5     # Should be uneven
padding = int((filter_size - 1) / 2)


##################
# Model training #
##################
class CNNClassLabels(nn.Module):
    def __init__(self, filter_size=filter_size, padding=padding, num_classes=len(labels)):
        super(CNNClassLabels, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=filter_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=filter_size, stride=1, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 54 * 36, 256)  # 32 channels of size 54 * 36 (due to 216x144 being pooled twice)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 54 * 36)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.image_folder = image_folder
        self.data = []
        for index, row in dataframe.iterrows():
            img_path = os.path.join(image_folder, f"{row['ID']}_thumb.jpg")
            label = row['Reviews per Month (Class)']
            self.data.append((img_path, label))

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)
        image = image.convert("RGB")
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        
        if torch.cuda.is_available():
            label = label.cuda(device=DEVICE)
            image = image.cuda() # Convert to tensor before cuda?

        return image, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((216, 144))
])

#df_sample = df.sample(n=1000, random_state=42)
df_sample = df

train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)

train_dataset = ImageDataset(train_df, image_folder=IMAGE_FOLDER, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = ImageDataset(test_df, image_folder=IMAGE_FOLDER, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

if torch.cuda.is_available():
    print("Using CUDA")
    model = CNNClassLabels().to(device=DEVICE)
    criterion = nn.CrossEntropyLoss().to(device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
else:
    print("Not using CUDA")
    model = CNNClassLabels()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create directory to save trained models in
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

num_epochs = 10

# Train the model
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        images, labels = data
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:    # print every X mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0
    
    dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), os.path.join(models_dir, f"cnn_classlabels_e{epoch+1}_d{dt_string}_s{len(train_dataset)}"))
