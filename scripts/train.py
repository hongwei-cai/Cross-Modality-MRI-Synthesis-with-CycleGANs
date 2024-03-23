# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from utils.dataset import FastMRIDataset
from torch.utils.data import DataLoader

def train_model(dataloader, model, criterion, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            inputs = batch['image_t1']
            targets = batch['image_t2']
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    print("This script is intended to be imported and used in main.py rather than executed directly.")
