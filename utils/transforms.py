# transforms.py
import torch
from torchvision import transforms


class ToTensor(object):
    """
    Converts NumPy arrays in a sample to PyTorch tensors.
    """

    def __call__(self, sample):
        """
        Applies the transformation to a sample.

        Args:
            sample (dict): A dictionary containing the image data.

        Returns:
            dict: A dictionary containing the transformed image data in PyTorch tensors.
        """

        image_t1, image_t2 = sample['image_t1'], sample['image_t2']

        # Handle differing channel dimensions between NumPy and PyTorch
        image_t1 = image_t1.transpose((2, 0, 1))  # Move channels to first dimension
        image_t2 = image_t2.transpose((2, 0, 1))

        # Convert NumPy arrays to PyTorch tensors
        return {
            'image_t1': torch.from_numpy(image_t1),
            'image_t2': torch.from_numpy(image_t2)
        }

# Add more transformations as needed, following the same pattern
