from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import torch
import random
from .sr_util import (
    get_paths_from_images,  # Update these helpers to scan PET-specific directories
    transform_augment       # Adjust augmentations to fit PET data requirements
)

class PETDataset(Dataset):
    def __init__(self, dataroot, img_size, split='train', data_len=-1):
        """
        Initialize the dataset for PET images.
        Args:
            dataroot (str): Root directory for PET image data.
            img_size (int): Target image size (assumes square resizing).
            split (str): 'train' or 'test' split.
            data_len (int): Number of images to include in the dataset (-1 for all images).
        """
        self.img_size = img_size
        self.data_len = data_len
        self.split = split
        
        # Load PET image paths
        self.img_ld_path, self.img_fd_path = get_paths_from_images(dataroot)  # Adjust for PET-specific folder structure
        self.data_len = len(self.img_ld_path)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        """
        Fetch and preprocess a single PET image pair (low-dose and full-dose).
        Args:
            index (int): Index of the image pair to load.
        Returns:
            dict: Dictionary containing preprocessed full-dose image (FD), low-dose image (LD),
                  and a unique identifier for the case (case_name).
        """
        img_FD = None
        img_LD = None
        case_name = None

        # Extract case name from the low-dose image path
        base_name = self.img_ld_path[index].split('/')[-1]
        case_name = base_name.split('_')[0]

        # Load and preprocess PET images
        img_LD = Image.open(self.img_ld_path[index]).convert("L")  # Convert to grayscale; use "RGB" if needed
        img_FD = Image.open(self.img_fd_path[index]).convert("L")  # Adjust preprocessing for PET

        # Resize images to the specified size
        img_LD = img_LD.resize((self.img_size, self.img_size))
        img_FD = img_FD.resize((self.img_size, self.img_size))

        # Apply augmentations and normalization
        [img_LD, img_FD] = transform_augment(
            [img_LD, img_FD], split=self.split, min_max=(-1, 1)  # Ensure normalization for PET-specific range
        )

        return {'FD': img_FD, 'LD': img_LD, 'case_name': case_name}
