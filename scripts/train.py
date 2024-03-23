# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet  # Assuming UNet is your model architecture
from utils.dataset import FastMRIDataset  # Assuming this is your custom dataset class
from torch.utils.data import DataLoader


def train_model(dataloader, model, criterion, optimizer, num_epochs=25):
    """
    Trains the provided model on the given dataset using the specified optimizer and loss criterion.

    Args:
        dataloader (DataLoader): A PyTorch DataLoader object providing batches of training data.
        model (nn.Module): The PyTorch model to be trained.
        criterion (nn.Module): The loss function used for calculating training loss.
        optimizer (optim.Optimizer): The optimizer used for updating model weights.
        num_epochs (int, optional): The number of epochs to train for (defaults to 25).
    """

    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')  # Print epoch information

        # Loop through each batch in the dataloader
        for i, batch in enumerate(dataloader):
            inputs = batch['image_t1']  # Extract input (T1 image)
            targets = batch['image_t2']  # Extract target (T2 image)

            # Clear optimizer gradients
            optimizer.zero_grad()

            # Forward pass: Get model outputs and calculate loss
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass: Propagate gradients
            loss.backward()

            # Optimizer step: Update model weights based on gradients
            optimizer.step()

            # Print training progress every 10 steps
            if (i + 1) % 10 == 0:
                print(f'Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

# Code for running the training script directly (usually not recommended)
if __name__ == "__main__":
    print("This script is intended to be imported and used in main.py rather than executed directly.")
