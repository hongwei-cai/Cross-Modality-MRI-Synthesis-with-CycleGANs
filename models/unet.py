# unet.py
import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    """
    Creates a sequence of two convolutional layers with ReLU activation 
    used as a building block in the U-Net architecture.

    Args:
        in_channels (int): Number of input channels for the first convolutional layer.
        out_channels (int): Number of output channels for the second convolutional layer.

    Returns:
        nn.Sequential: A PyTorch sequential module containing the two convolutional layers and ReLU activations.
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """
    A PyTorch implementation of the U-Net architecture for image segmentation.

    Args:
        n_channels (int): Number of input channels (e.g., 1 for grayscale images).
        n_classes (int): Number of output channels (e.g., number of classes for segmentation).
    """

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()

        # Encoder (downward path)
        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        # Pooling layers for downsampling
        self.maxpool = nn.MaxPool2d(2)

        # Upsampling layers for expanding features
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decoder (upward path)
        self.dconv_up3 = double_conv(256 + 512, 256)  # Concatenated feature channels
        self.dconv_up2 = double_conv(128 + 256, 128)  # Concatenated feature channels
        self.dconv_up1 = double_conv(64 + 128, 64)  # Concatenated feature channels

        # Final convolutional layer for output
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        """
        Performs the forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input tensor containing the image data.

        Returns:
            torch.Tensor: Output tensor representing the model's predictions.
        """

        # Encoder path with downsampling and feature extraction
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        # Decoder path with upsampling and feature concatenation
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)  # Concatenate features from encoder

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)  # Concatenate features from encoder

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)  # Concatenate features from encoder

        x = self.dconv_up1(x)

        # Final output layer
        out = self.conv_last(x)

        return out
