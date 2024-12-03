import os
from PIL import Image
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
        self.image_paths = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert("L")  # Load as grayscale

        # Split the image into LPET and FDPET
        width, height = img.size
        lpet = img.crop((0, 0, width // 2, height))
        fdpet = img.crop((width // 2, 0, width, height))

        # Resize and normalize
        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        lpet = transform(lpet)
        fdpet = transform(fdpet)

        return {'LPET': lpet, 'FDPET': fdpet, 'case_name': os.path.basename(img_path)}
