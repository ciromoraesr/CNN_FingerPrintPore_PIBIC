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
from architecture import PoreDetectionCNN2
from process import FingerprintData

class EnhancedPoreDetectionCNN(nn.Module):
    def __init__(self, in_channels=1):
        super(EnhancedPoreDetectionCNN, self).__init__()
        # Encoder path with residual connections
        # Initial block with higher filter count for better feature extraction
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        
        # Down-sampling
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second block with more filters
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        
        # Down-sampling
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third block with even more filters
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        
        # Down-sampling
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottom block
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        
        # Decoder path with transposed convolutions
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        
        # Skip connection handling
        self.conv3_3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # 128 + 128 = 256 input channels
        self.bn3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.conv2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  # 64 + 64 = 128 input channels
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_4 = nn.BatchNorm2d(64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.conv1_3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # 32 + 32 = 64 input channels
        self.bn1_3 = nn.BatchNorm2d(32)
        self.conv1_4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_4 = nn.BatchNorm2d(32)
        
        # Final output layer
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
        # Dilated convolutions for multi-scale context
        self.dil_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=2, dilation=2)
        self.dil_bn1 = nn.BatchNorm2d(256)
        self.dil_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=4, dilation=4)
        self.dil_bn2 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        # Save input dimensions for later use
        _, _, h, w = x.size()
        
        # Encoder path
        # First block
        c1 = F.relu(self.bn1_1(self.conv1_1(x)))
        c1 = F.relu(self.bn1_2(self.conv1_2(c1)))
        p1 = self.pool1(c1)
        
        # Second block
        c2 = F.relu(self.bn2_1(self.conv2_1(p1)))
        c2 = F.relu(self.bn2_2(self.conv2_2(c2)))
        p2 = self.pool2(c2)
        
        # Third block
        c3 = F.relu(self.bn3_1(self.conv3_1(p2)))
        c3 = F.relu(self.bn3_2(self.conv3_2(c3)))
        p3 = self.pool3(c3)
        
        # Bottom block with dilated convolutions for multi-scale context
        c4 = F.relu(self.bn4_1(self.conv4_1(p3)))
        c4 = F.relu(self.bn4_2(self.conv4_2(c4)))
        
        # Add dilated convolutions for capturing multi-scale features
        d1 = F.relu(self.dil_bn1(self.dil_conv1(c4)))
        d2 = F.relu(self.dil_bn2(self.dil_conv2(c4)))
        c4 = c4 + d1 + d2  # Residual connection
        c4 = self.dropout(c4)
        
        # Decoder path - use interpolation instead of transposed convolutions
        # This resolves size mismatches that might occur with transposed convolutions
        u3 = F.interpolate(c4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        u3 = self.upconv3(c4)  # Apply convolution after interpolation for channel reduction
        
        # Handle possible size mismatch before concatenation
        if u3.shape[2:] != c3.shape[2:]:
            u3 = F.interpolate(u3, size=c3.shape[2:], mode='bilinear', align_corners=False)
            
        u3 = torch.cat([u3, c3], dim=1)
        c3 = F.relu(self.bn3_3(self.conv3_3(u3)))
        c3 = F.relu(self.bn3_4(self.conv3_4(c3)))
        c3 = self.dropout(c3)
        
        u2 = F.interpolate(c3, size=c2.shape[2:], mode='bilinear', align_corners=False)
        u2 = self.upconv2(c3)  # Apply convolution after interpolation
        
        # Handle possible size mismatch before concatenation
        if u2.shape[2:] != c2.shape[2:]:
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='bilinear', align_corners=False)
            
        u2 = torch.cat([u2, c2], dim=1)
        c2 = F.relu(self.bn2_3(self.conv2_3(u2)))
        c2 = F.relu(self.bn2_4(self.conv2_4(c2)))
        
        u1 = F.interpolate(c2, size=c1.shape[2:], mode='bilinear', align_corners=False)
        u1 = self.upconv1(c2)  # Apply convolution after interpolation
        
        # Handle possible size mismatch before concatenation
        if u1.shape[2:] != c1.shape[2:]:
            u1 = F.interpolate(u1, size=c1.shape[2:], mode='bilinear', align_corners=False)
            
        u1 = torch.cat([u1, c1], dim=1)
        c1 = F.relu(self.bn1_3(self.conv1_3(u1)))
        c1 = F.relu(self.bn1_4(self.conv1_4(c1)))
        
        # Ensure output has the same spatial dimensions as input
        if c1.shape[2:] != (h, w):
            c1 = F.interpolate(c1, size=(h, w), mode='bilinear', align_corners=False)
        
        # Final output with sigmoid activation
        out = self.final(c1)
        return torch.sigmoid(out)


# A simpler attention mechanism that's more robust to size mismatches
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_out)
        
        return self.sigmoid(x_out) * x

# Improved accuracy function with F1 score
def compute_enhanced_metrics(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Calculate true positives, false positives, false negatives
    tp = torch.sum((pred_binary == 1) & (target_binary == 1)).float()
    fp = torch.sum((pred_binary == 1) & (target_binary == 0)).float()
    fn = torch.sum((pred_binary == 0) & (target_binary == 1)).float()
    tn = torch.sum((pred_binary == 0) & (target_binary == 0)).float()
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)
    
    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Calculate Intersection over Union (IoU)
    iou = tp / (tp + fp + fn + 1e-10)
    
    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item()
    }


# Loss function combining Binary Cross Entropy and Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCELoss()
    
    def forward(self, inputs, targets):
        # BCE Loss
        bce = self.bce_loss(inputs, targets)
        
        # Dice Loss
        smooth = 1.0
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        dice = 1 - (2. * intersection + smooth) / (union + smooth)
        
        # Combined loss
        return self.bce_weight * bce + (1 - self.bce_weight) * dice
    
def evaluate(model, loader, size, device, criterion):
    model.eval()
    loss, acc = 0.0, 0.0
    metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            # Handle different output formats based on model type
            if isinstance(model, PoreDetectionCNN2):
                batch_loss = criterion(outputs, targets)
                batch_acc = compute_accuracy(outputs, targets)
                # Original model doesn't have additional metrics
            else:  # EnhancedPoreDetectionCNN
                # For enhanced model, outputs are already sigmoid-activated
                # Convert to logits for BCE loss if needed
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    batch_loss = criterion(torch.log(outputs / (1 - outputs + 1e-7)), targets)
                else:
                    batch_loss = criterion(outputs, targets)
                
                # Compute all metrics
                batch_metrics = compute_enhanced_metrics(outputs, targets)
                batch_acc = batch_metrics['accuracy']
                
                # Accumulate additional metrics
                for key in metrics:
                    metrics[key] += batch_metrics[key] * images.size(0)
            
            loss += batch_loss.item() * images.size(0)
            acc += batch_acc * images.size(0)
    
    # Normalize metrics
    for key in metrics:
        metrics[key] /= size
    
    return loss/size, acc/size, metrics


def train_model(train_loader, val_loader, train_size, val_size, date_today, model_type='enhanced', num_epochs=50, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_iou': []
    }

    # Model selection based on specified type
    if model_type == 'enhanced':
        model = EnhancedPoreDetectionCNN().to(device)
        # Use combined loss for enhanced model
        criterion = CombinedLoss(bce_weight=0.5)
    else:  # 'original'
        model = PoreDetectionCNN2().to(device)
        pos_weight = torch.tensor([10.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    criterion.to(device)
    best_val_loss = float('inf')
    model_name = f"best_{model_type}_model_{date_today}.pth"
    save_path = "model_folder/" + model_name
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        train_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle different output formats based on model type
            if isinstance(model, PoreDetectionCNN2):
                loss = criterion(outputs, targets)
                acc = compute_accuracy(outputs, targets)
            else:  # EnhancedPoreDetectionCNN
                # For enhanced model, outputs are already sigmoid-activated
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    loss = criterion(torch.log(outputs / (1 - outputs + 1e-7)), targets)
                else:
                    loss = criterion(outputs, targets)
                
                batch_metrics = compute_enhanced_metrics(outputs, targets)
                acc = batch_metrics['accuracy']
                
                # Accumulate additional metrics
                for key in train_metrics:
                    train_metrics[key] += batch_metrics[key] * images.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_acc += acc * images.size(0)

        val_loss, val_acc, val_metrics = evaluate(model, val_loader, val_size, device, criterion)

        scheduler.step(val_loss)

        train_loss /= train_size
        train_acc /= train_size
        
        # Normalize training metrics
        for key in train_metrics:
            train_metrics[key] /= train_size

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Store additional metrics
        for key in val_metrics:
            history[f'val_{key}'].append(val_metrics[key])

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
        
        # Print additional metrics if using enhanced model
        if model_type == 'enhanced':
            print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f} | Val IoU: {val_metrics['iou']:.4f}")

    return history


