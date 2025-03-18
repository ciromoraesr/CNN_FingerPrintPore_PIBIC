import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from process import FingerprintData



class PoreDetectionCNN(nn.Module):
    def __init__(self):
        super(PoreDetectionCNN, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(8)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(8)

        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(16)

        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout2d(0.5)
        self.pointwise_conv = nn.Conv2d(56, 1, kernel_size=1)  

    def forward(self, x):
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1))) + x1

        x2 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2))) + x2

        x3 = F.relu(self.bn3_1(self.conv3_1(x2)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3))) + x3


        
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_final = self.pointwise_conv(x_concat)  
        return (torch.sigmoid(x_final))
    

class PoreDetectionCNN2(nn.Module):
    def __init__(self):
        super(PoreDetectionCNN2, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(16)

        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(32)

        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)

        self.dropout = nn.Dropout2d(0.5)
        self.pointwise_conv = nn.Conv2d(112, 1, kernel_size=1)  

    def forward(self, x):
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1))) + x1

        x2 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2))) + x2

        x3 = F.relu(self.bn3_1(self.conv3_1(x2)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3))) + x3


        
        x_concat = torch.cat([x1, x2, x3], dim=1)
        x_final = self.pointwise_conv(x_concat)  
        return (torch.sigmoid(x_final))
    
def compute_accuracy(pred, target, threshold=0.5):  
    pred_probs = torch.sigmoid(pred)  
    correct = (torch.abs(pred_probs - target) < threshold).float()
    return correct.mean().item()


def evaluate(model, loader, size, device, criterion):
    model.eval()
    loss, acc,  = 0.0, 0.0
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            batch_loss = criterion(outputs, targets)
            batch_acc = compute_accuracy(outputs, targets)
            
            loss += batch_loss.item() * images.size(0)
            acc += batch_acc * images.size(0)
    
    return loss/size, acc/size


def train_model(train_loader, val_loader, train_size, val_size, date_today, num_epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        
    }

    model = PoreDetectionCNN2().to(device)
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion.to(device)
    best_val_loss = float('inf')
    model_name = "best_model_" + date_today + ".pth"
    save_path = "model_folder/" + model_name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            acc = compute_accuracy(outputs, targets)
            train_loss += loss.item() * images.size(0)
            train_acc += acc * images.size(0)

        val_loss, val_acc,  = evaluate(model, val_loader, val_size, device, criterion)

        scheduler.step(val_loss)

        train_loss /= train_size
        train_acc /= train_size

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, save_path)
            print(f"Model saved at epoch {epoch + 1}")

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return history

