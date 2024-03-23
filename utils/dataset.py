# dataset.py
import torch
from torch.utils.data import Dataset
import h5py
import os


class FastMRIDataset(Dataset):
    """
    A PyTorch dataset class for loading FastMRI data from HDF5 files.
    """

    def __init__(self, root_dir, transform=None):
        # Store parameters
        self.root_dir = root_dir
        self.transform = transform

        # Get a list of image files in the directory
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Handle tensor indices (if applicable)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the full path to the image file
        img_name = os.path.join(self.root_dir, self.file_list[idx])

        # Open the HDF5 file
        with h5py.File(img_name, 'r') as file:
            # Load the T1 and T2 images from the HDF5 file (adjust keys as needed)
            image_t1 = file['t1'][:]  # Access the 't1' dataset within the file
            image_t2 = file['t2'][:]  # Access the 't2' dataset within the file

        # Create a sample dictionary
        sample = {'image_t1': image_t1, 'image_t2': image_t2}

        # Apply optional transformations (if provided)
        if self.transform:
            sample = self.transform(sample)

        return sample
