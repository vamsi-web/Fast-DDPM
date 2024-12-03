import os
import torch
import torchvision
import numpy as np
from PIL import Image
import glob
import random

# Supported image extensions
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.mat']

def is_image_file(filename):
    """
    Check if a file is an image based on its extension.
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_valid_paths_from_images(path):
    """
    Get valid image paths from the given directory.
    Args:
        path (str): Directory containing images.
    Returns:
        list: List of valid image file paths.
    """
    assert os.path.isdir(path), f"{path} is not a valid directory"
    images = []
    for dirpath, _, fnames in os.walk(path):
        fnames = sorted([f for f in fnames if is_image_file(f)])
        images.extend(os.path.join(dirpath, f) for f in fnames)
    assert images, f"No valid images found in {path}"
    return images

def get_paths_from_images(path):
    """
    Get all valid image paths in a given directory.
    Args:
        path (str): The root directory containing images.
    Returns:
        list: Sorted list of image file paths.
    """
    assert os.path.isdir(path), f"{path} is not a valid directory"
    images = glob.glob(os.path.join(path, "**", "*.png"), recursive=True)
    assert images, f"No valid image files found in {path}"
    return sorted(images)

def get_valid_paths_from_test_images(path):
    """
    Get valid test image paths from the given directory, excluding invalid files.
    Args:
        path (str): Directory containing test images.
    Returns:
        list: List of valid test image file paths.
    """
    assert os.path.isdir(path), f"{path} is not a valid directory"
    images = []
    for dirpath, _, fnames in os.walk(path):
        fnames = sorted([f for f in fnames if is_image_file(f)])
        images.extend(os.path.join(dirpath, f) for f in fnames)
    assert images, f"No valid test images found in {path}"
    return images

def transform2numpy(img):
    """
    Convert an image to a NumPy array and normalize to [0, 1].
    """
    img = np.array(img)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # Handle images with more than 3 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def transform2tensor(img, min_max=(0, 1)):
    """
    Convert a NumPy array to a PyTorch tensor and normalize to a specified range.
    """
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img

# Data augmentation and transformations
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
Resize = torchvision.transforms.Resize((128, 128), antialias=True)

def transform_augment(img_list, split='val', min_max=(-1, 1)):
    """
    Apply transformations and augmentations to a list of images.
    Args:
        img_list (list): List of images to transform.
        split (str): Dataset split ('train' or 'val').
        min_max (tuple): Range for normalization.
    Returns:
        list: Transformed images.
    """
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img
