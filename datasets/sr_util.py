import os
import torch
import torchvision
import numpy as np
from PIL import Image
import glob
import random

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.mat']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# Function to get image paths for PET data
def get_paths_from_images(path):
    """
    Get paths for PET images (size: 128x256) containing both LPET and FDPET.
    """
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    
    pet_images = glob.glob(path + "**/*.png", recursive=True)
    assert pet_images, '{:s} has no valid PET image files'.format(path)
    return sorted(pet_images)

# Function to split 128x256 PET images into LPET and FDPET
def split_pet_image(pet_image):
    """
    Split a 128x256 PET image into LPET (left half) and FDPET (right half).
    Args:
        pet_image (PIL.Image): The combined PET image.
    Returns:
        tuple: LPET image and FDPET image as PIL.Image objects.
    """
    width, height = pet_image.size  # Expecting 256x128
    assert width == 256 and height == 128, "Image must be 128x256 in size."
    
    # Crop left (LPET) and right (FDPET) halves
    lpet = pet_image.crop((0, 0, 128, 128))   # Left half
    fdpet = pet_image.crop((128, 0, 256, 128))  # Right half
    return lpet, fdpet

# Transform images to NumPy format
def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.  # Normalize to [0, 1]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # Handle 3+ channel images, keep only the first 3 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

# Convert to tensor and normalize to a specified range
def transform2tensor(img, min_max=(0, 1)):
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img

# Apply augmentations
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
Resize = torchvision.transforms.Resize((128, 128), antialias=True)

def transform_augment(img_list, split='val', min_max=(-1, 1)):
    """
    Apply transformations and augmentations to a list of images.
    """
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

# Dataset loader for PET images
class PETDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, img_size=128, split='train', data_len=-1):
        """
        Initialize the dataset for PET images.
        Args:
            dataroot (str): Root directory containing PET images.
            img_size (int): Target size for resized images.
            split (str): 'train' or 'val' split.
            data_len (int): Number of images to use (-1 for all images).
        """
        self.img_size = img_size
        self.split = split
        self.pet_paths = get_paths_from_images(dataroot)
        self.data_len = len(self.pet_paths) if data_len == -1 else data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        Args:
            index (int): Index of the image.
        Returns:
            dict: Dictionary containing LPET, FDPET, and metadata.
        """
        pet_image = Image.open(self.pet_paths[index]).convert("L")  # Load grayscale image
        lpet, fdpet = split_pet_image(pet_image)  # Split into LPET and FDPET

        # Resize images
        lpet = lpet.resize((self.img_size, self.img_size))
        fdpet = fdpet.resize((self.img_size, self.img_size))

        # Apply augmentations and transformations
        lpet, fdpet = transform_augment([lpet, fdpet], split=self.split, min_max=(-1, 1))

        # Extract metadata
        case_name = os.path.basename(self.pet_paths[index]).split('.')[0]

        return {'LPET': lpet, 'FDPET': fdpet, 'case_name': case_name}
