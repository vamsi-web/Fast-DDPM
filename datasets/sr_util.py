import os
import torch
import torchvision
import numpy as np
from PIL import Image
import glob

# Supported image extensions
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.mat']

# Helper function to check if a file is an image
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# Function to get all valid image paths
def get_valid_paths_from_images(path):
    assert os.path.isdir(path), f"{path} is not a valid directory"
    images = []
    for dirpath, _, fnames in os.walk(path):
        fnames = sorted([f for f in fnames if is_image_file(f)])
        images.extend(os.path.join(dirpath, f) for f in fnames)
    assert images, f"No valid images found in {path}"
    return images

# Function to get image paths recursively
def get_paths_from_images(path):
    assert os.path.isdir(path), f"{path} is not a valid directory"
    images = glob.glob(os.path.join(path, "**", "*.png"), recursive=True)
    assert images, f"No valid image files found in {path}"
    return sorted(images)

# Function to get valid test image paths
def get_valid_paths_from_test_images(path):
    assert os.path.isdir(path), f"{path} is not a valid directory"
    images = []
    for dirpath, _, fnames in os.walk(path):
        fnames = sorted([f for f in fnames if is_image_file(f)])
        images.extend(os.path.join(dirpath, f) for f in fnames)
    assert images, f"No valid test images found in {path}"
    return images

# Function to get paths from NumPy files
def get_paths_from_npys(path_data, path_gt):
    """
    Get paths for NumPy files representing data and ground truth.
    Args:
        path_data (str): Path to directory containing data NumPy files.
        path_gt (str): Path to directory containing ground truth NumPy files.
    Returns:
        tuple: (list of data NumPy file paths, list of GT NumPy file paths).
    """
    assert os.path.isdir(path_data), f"{path_data} is not a valid directory"
    assert os.path.isdir(path_gt), f"{path_gt} is not a valid directory"

    data_npy = glob.glob(os.path.join(path_data, "*.npy"))
    gt_npy = glob.glob(os.path.join(path_gt, "*.npy"))

    assert data_npy, f"No valid data NumPy files found in {path_data}"
    assert gt_npy, f"No valid GT NumPy files found in {path_gt}"
    assert len(data_npy) == len(gt_npy), "Mismatch between data and GT NumPy files"

    return sorted(data_npy), sorted(gt_npy)

# Transform an image to a NumPy array and normalize
def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

# Convert a NumPy array to a PyTorch tensor with normalization
def transform2tensor(img, min_max=(0, 1)):
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img

# Augmentation and transformation utilities
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()

def transform_augment(img_list, split='val', min_max=(-1, 1)):
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

# BRATS-specific transformation
def brats_transform_augment(img_list, split='val'):
    imgs = [totensor(img) for img in img_list]
    ret_img = [img.clamp(-1.0, 1.0) for img in imgs]
    return ret_img
