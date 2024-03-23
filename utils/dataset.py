# dataset.py
import torch
from torch.utils.data import Dataset
import h5py  # If your data is stored in HDF5 files
import os

class FastMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_list[idx])
        with h5py.File(img_name, 'r') as file:
            image_t1 = file['t1'][:]  # Adjust based on your data structure
            image_t2 = file['t2'][:]  # Adjust based on your data structure
        sample = {'image_t1': image_t1, 'image_t2': image_t2}

        if self.transform:
            sample = self.transform(sample)

        return sample
