from torchvision.transforms.functional import adjust_brightness as torch_adjust_brightness
from torchvision.transforms.functional import adjust_contrast as torch_adjust_contrast
from torchvision.transforms.functional import adjust_gamma as torch_adjust_gamma
from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms.functional import hflip as torch_hflip, vflip as torch_vlip
from torchvision.transforms.functional import resize as torch_resize
from torchvision.transforms import RandomAffine

import numpy as np

import random 

from PIL import ImageFilter

from timeit import default_timer as timer

def adjust_brightness(image, min_brightness_ratio = 0.5, max_brightness_ratio = 1.5, variance = 0.5):
    normal_sample = np.random.normal(0, variance)
    if normal_sample > 0:
        brightness_scale = 1 + normal_sample
    else:
        brightness_scale = 1.0 / (1 + np.abs(normal_sample))
    brightness_scale = np.clip(brightness_scale, min_brightness_ratio, max_brightness_ratio)
    image = torch_adjust_brightness(image, brightness_scale)
    return image   

def adjust_contrast(image, min_contrast_ratio = 0.5, max_contrast_ratio = 1.5, variance = 0.5):
    normal_sample = np.random.normal(0, variance)
    if normal_sample > 0:
        contrast_scale = 1 + normal_sample
    else:
        contrast_scale = 1.0 / (1 + np.abs(normal_sample))
    contrast_scale = np.clip(contrast_scale, min_contrast_ratio, max_contrast_ratio)
    image = torch_adjust_contrast(image, contrast_scale)
    return image 

def adjust_gamma(image, min_gamma = 0.5, max_gamma = 2, variance = 0.5):
    normal_sample = np.random.normal(0, variance)
    if normal_sample > 0:
        gamma = 1 + normal_sample
    else:
        gamma = 1.0 / (1 + np.abs(normal_sample))
    gamma = np.clip(gamma, min_gamma, max_gamma)
    image = torch_adjust_gamma(image, gamma)
    return image   

def uniform_rotate(image, mask, min_angle = -10, max_angle = 10):
    rotate_angle = np.random.uniform(min_angle, max_angle)
    image = torch_rotate(image, rotate_angle)
    mask = torch_rotate(mask, rotate_angle)
    return image, mask

def rotate_90(image, mask):
    rotate_angle = np.random.randint(1, 4) * 90
    image = torch_rotate(image, rotate_angle)
    mask = torch_rotate(mask, rotate_angle)
    return image, mask

def hflip(image, mask):
    image = torch_hflip(image)
    mask = torch_hflip(mask)
    return image, mask

def vflip(image, mask):
    image = torch_vflip(image)
    mask = torch_vflip(mask)
    return image, mask

def resize(image, mask, image_dim):
    image = torch_resize(image, image_dim)
    mask = torch_resize(mask, image_dim)
    return image, mask

def random_square_crop_by_scale(image, mask, scale = 0.9):
    new_size = int(image.shape[1] * scale)
    start_x = np.random.randint(0, image.shape[1] - new_size)
    start_y = np.random.randint(0, image.shape[1] - new_size)
    image_cropped = image[:, start_x: start_x + new_size, start_y: start_y + new_size]
    mask_cropped = mask[:, start_x: start_x + new_size, start_y: start_y + new_size]
    return image_cropped, mask_cropped

def gaussian_blur(image, radius = 2):
    blur = ImageFilter.GaussianBlur(radius)
    image = ImageFilter.filter(image)
    return image

def shear(image, mask, shear_degrees):
    affine = RandomAffine(degrees = 0,shear = shear_degrees)
    seed = int(timer())
    
    random.seed(seed)
    image = affine(image)
    random.seed(seed)
    mask = affine(mask)
    
    return image, mask