import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PETDataset(Dataset):
    def __init__(self, dataroot, img_size=128, split='train'):
        """
        Args:
            dataroot (str): Path to the dataset directory.
            img_size (int): Target size for resizing images.
            split (str): Either 'train' or 'test'.
        """
        self.img_size = img_size
        self.split = split
        self.file_paths = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.endswith('.mat')]

        if not self.file_paths:
            raise ValueError(f"No .mat files found in directory: {dataroot}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        mat_path = self.file_paths[index]
        mat_data = sio.loadmat(mat_path)  # Load .mat file
        #if 'image' not in mat_data:
            #raise KeyError(f"'image' key not found in {mat_path}")

        # Assuming the data is stored under the key 'image'
        img = mat_data['img']  # Replace 'image' with the appropriate key if different
        if img.ndim == 2:  # Ensure it's 3D (H, W, C)
            img = np.expand_dims(img, axis=-1)

        # Split the image into LPET and FDPET
        h, w, c = img.shape
        if w != 256 or c != 1:
            raise ValueError(f"Expected image shape (H, 256, 1), but got {img.shape}")

        lpet = img[:, :128]  # Left half (LPET)
        fdpet = img[:, 128:]  # Right half (FDPET)

        # Resize and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        lpet = transform(lpet)
        fdpet = transform(fdpet)

        return {'LPET': lpet, 'FDPET': fdpet, 'case_name': os.path.basename(mat_path)}
