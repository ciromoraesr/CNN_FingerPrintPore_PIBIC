# -- coding: utf-8 --


import os
import shutil

try:
    os.mkdir("rep")
except:
    print("ja existe")
src = r"L3SF_V2/Pore ground truth/"


origin1 = os.path.join(src,(os.listdir(src)[1]))
origin2 = os.path.join(src,(os.listdir(src)[2]))
imgs = []
labels = []
for i in range(5):
    char = "R" + str(i+1)
    imgs.append(os.listdir(os.path.join(origin1, char)))
    labels.append(os.listdir(os.path.join(origin2, char)))
    
#%%
c = "carro.carro"
print(type(c.split(".")[0]))
os.rename(c, "aff")
#%%
import os
import shutil

try:
    os.mkdir("rep/images")
    os.mkdir("rep/labels")
except:
    print("ja existe")

destimg = r"rep/images"
destlbl = r"rep/labels"

count = 0

for i in range(5):
    char = "R" + str(i + 1)
    img_path = os.path.join(origin1, char)
    lbl_path = os.path.join(origin2, char)

    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        continue

    for file in os.listdir(img_path):
        img_file_path = os.path.join(img_path, file)
        if not os.path.isfile(img_file_path):
            print(f"Skipping non-file: {file}")
            continue

        temp, ext = os.path.splitext(file)
        if count > 147:
            new_name = f"{temp}_{count}{ext}"
        else:
            new_name = f"{temp}{ext}"

        img_dest = os.path.join(destimg, new_name)
        lbl_dest = os.path.join(destlbl, new_name.replace(ext, ".tsv"))  # Assuming labels have .txt extension

        try:
            shutil.move(img_file_path, img_dest)
        except Exception as e:
            print(f"Error moving image {file}: {e}")

        lbl_file_path = os.path.join(lbl_path, temp + ".tsv")  # Match label file
        if os.path.exists(lbl_file_path):
            try:
                shutil.move(lbl_file_path, lbl_dest)
            except Exception as e:
                print(f"Error moving label {file}: {e}")

        count += 1

count = 0
for i in range(5):
    char = "R" + str(i + 1)
    path = os.path.join(origin2, char)
    
    if not os.path.exists(path):
       
        continue

    
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if not os.path.isfile(file_path):
            print(f"Skipping non-file: {file}")
            continue
        
        
        temp, ext = os.path.splitext(file)
        if count > 147:
            temp = f"{temp}_{count}{ext}"
        else:
            temp = f"{temp}{ext}"
        
        dest = os.path.join(destlbl, temp)
        try:
            shutil.move(file_path, dest)
        except Exception as e:
            print(f"{e}")
        
        count += 1   
#%%
print(len(os.listdir("rep/images")), len(os.listdir("rep/labels")))
#%%
import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FingeprintData(Dataset):
    
    def __init__(self, image_dir, label_dir, transform = None, sigma = 5):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.sigma = sigma
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg'))]
        self.label_files = [f.replace(os.path.splitext(f)[1], 'tsv') for f in self.image_files]
        for lf in self.label_files:
            assert os.path.isfile(os.path.join(label_dir, lf)), f"Missing: {lf}"
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = self.label_files[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')
        image = transforms.ToTensor()(image)
        
        label_path = os.path.join(self.label_dir, label_name)
        coords = pd.read_csv(label_path, sep = '\t', names = ['x', 'y']).values
        
        h, w = image.shpae[1], image.shape[2]
        heatmap = self._create_heatmap(coords, (h,w))
        heatmap = torch.tensor(heatmap, dtype = torch.float32).unsqueeze(0)
        
        return image, heatmap


    def _create_heatmap(self, coords, img_size):
        heatmap = np.zeros(img_size, dtype = np.float32)
        h, w = img_size
        window_size = int(6*self.sigma) + 1
        radius = window_size // 2
        
        for x, y in coords:
            x, y = float(x), float(y)
            x_min = max(0, int(x - radius))
            x_max = min(w, int(x + radius + 1))
            y_min = max(0, int(y - radius))
            y_max = min(h, int(y + radius + 1))
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))
            heatmap[y_min:y_max, x_min:x_max] += gaussian

        heatmap = np.clip(heatmap, 0, 1)
        return heatmap

#%%
#adaptar
import torch.nn as nn

class PoreDetectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
            )
    def forward(self, x):
        return self.layers(x)
#%%


def train_model(img_dir, lbl_dir, num_epochs = 50, batch_size = 8, lr = 0.01):
    device = torch.device('cuda' if torch.cuda .is_available() else 'cpu')
    dataset = FingerprintData(img_dir, lbl_dir)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    
    model = PoreDetectionCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_empochs):
        model.train()
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            
            optimizer.zero_grad()
            
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            
            optimizer.step()
            
            running_loss += loss.item() + images.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f'Epoch: {epoch+1}/{num_epochs}, LOSS: {epoch_loss:.4f}')
