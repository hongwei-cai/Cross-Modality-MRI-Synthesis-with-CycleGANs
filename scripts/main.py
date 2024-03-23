# main.py
from models.unet import UNet
from utils.dataset import FastMRIDataset
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def main():
    # Initialize dataset and dataloader
    train_dataset = FastMRIDataset(root_dir='../data/train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Model initialization
    n_channels = 1
    n_classes = 1
    model = UNet(n_channels, n_classes)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Starting training...")
    train_model(train_dataloader, model, criterion, optimizer, num_epochs=25)
    
    # Optionally, evaluate the model
    # val_dataset = FastMRIDataset(root_dir='path/to/your/dataset/val')
    # val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    # print("Evaluating model...")
    # evaluate_model(val_dataloader, model)

if __name__ == "__main__":
    main()
