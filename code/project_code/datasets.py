import torch
from torch.utils import data
from skimage.transform import AffineTransform, rotate
import torchvision
from PIL import Image
import os
import numpy as np
from utils import *
from augmentations import *
import matplotlib.pyplot as plt


class Promise2012_Dataset(data.Dataset):

    
    def __init__(self, dir_images, dir_masks, patients, augment=False, gpu=False, image_dim=(256, 256)):
        self.dir_images = dir_images
        self.dir_masks = dir_masks

        images = os.listdir(self.dir_images)
        masks = os.listdir(self.dir_masks)

        self.all_patients_images = sorted(
            [image for image in images if int(image[4:6]) in patients])
        self.all_patients_masks = sorted(
            [mask for mask in masks if int(mask[4:6]) in patients])

        self.augment = augment
        self.gpu = gpu
        self.image_dim = image_dim

    def __len__(self):
        return len(self.all_patients_images)


    def __getitem__(self, index):
        image_filename, mask_filename = self.all_patients_images[
            index], self.all_patients_masks[index]
        image, mask = Image.open(os.path.join(self.dir_images, image_filename)), \
            Image.open(os.path.join(self.dir_masks, mask_filename))

        to_tensor = torchvision.transforms.ToTensor()
        to_image = torchvision.transforms.ToPILImage()

        if self.augment:

            image, mask = to_tensor(image), to_tensor(mask)
            scale = np.random.uniform(0.6, 1.0)
            image, mask = random_square_crop_by_scale(image, mask, scale)
            image, mask = to_image(image), to_image(mask)

            image = adjust_brightness(image, 0.5, 2, variance = 0.5)
            image = adjust_contrast(image, 0.5, 2, variance = 0.5)            
            #image = adjust_gamma(image, min_gamma = 0.5, max_gamma = 2, variance = 0.5)
            
            if np.random.uniform(0, 1) < 0.5:
                image, mask = uniform_rotate(image, mask, -30, 30)
            else:
                image, mask = shear(image, mask, 30)
                
            if np.random.uniform(0, 1) < 0.2:
                image, mask = rotate_90(image, mask)

            if np.random.uniform(0, 1) < 0.3:
                image, mask = hflip(image, mask)

        if image.size[0] != self.image_dim[0] or image.size[1] != self.image_dim[1]:
            image, mask = resize(image, mask, self.image_dim)

        image, mask = to_tensor(image), to_tensor(mask)
        return image, mask, image_filename

################################## pixel-wise

class Promise2012_Dataset_Pixelwise(data.Dataset):

    
    def __init__(self, dir_images, dir_masks, patients, augment=False, gpu=False, image_dim=[(25,25), (51,51), (75,75)]):
        self.dir_images = dir_images
        self.dir_masks = dir_masks

        images = os.listdir(self.dir_images)
        masks = os.listdir(self.dir_masks)

        self.all_patients_images = sorted(
            [image for image in images if int(image[4:6]) in patients])
        self.all_patients_masks = sorted(
            [mask for mask in masks if int(mask[4:6]) in patients])

        self.augment = augment
        self.gpu = gpu
        self.image_dims = image_dim
        
        self.size = 256
        self.patches_per_image = (self.size-self.image_dims[-1][0])*(self.size-self.image_dims[-1][1])
        
    def __len__(self):
        return len(self.all_patients_images)*self.patches_per_image

    def __getitem__(self, index):
        index_image = index // self.patches_per_image
        image_filename, mask_filename = self.all_patients_images[
            index_image], self.all_patients_masks[index_image]
        image, mask = Image.open(os.path.join(self.dir_images, image_filename)), \
            Image.open(os.path.join(self.dir_masks, mask_filename))
        #print image_filename, index_image
        
        to_tensor = torchvision.transforms.ToTensor()
        to_image = torchvision.transforms.ToPILImage()

        if self.augment:

            image, mask = to_tensor(image), to_tensor(mask)
            scale = np.random.uniform(0.6, 1.0)
            image, mask = random_square_crop_by_scale(image, mask, scale)
            image, mask = to_image(image), to_image(mask)

            image = adjust_brightness(image, 0.5, 2, variance = 0.5)
            image = adjust_contrast(image, 0.5, 2, variance = 0.5)            
            #image = adjust_gamma(image, min_gamma = 0.5, max_gamma = 2, variance = 0.5)
            
            if np.random.uniform < 0.5:
                image, mask = uniform_rotate(image, mask, -30, 30)
            else:
                image, mask = shear(image, mask, 30)
                
            if np.random.uniform(0, 1) < 0.2:
                image, mask = rotate_90(image, mask)

            if np.random.uniform(0, 1) < 0.3:
                image, mask = hflip(image, mask)

        image, mask = resize(image, mask, (self.size, self.size))
        image, mask = to_tensor(image), to_tensor(mask)
        
        n, m = self.image_dims[-1][0] // 2, self.image_dims[-1][1] // 2
        n2, m2 = self.image_dims[-2][0] // 2, self.image_dims[-2][1] // 2
        n3, m3 = self.image_dims[-3][0] // 2, self.image_dims[-3][1] // 2

        index_x = (index % self.patches_per_image) % (self.size - self.image_dims[-1][1]) + n
        index_y = (index % self.patches_per_image) // (self.size - self.image_dims[-1][1]) + m
        #print index_x, index_y
        mask = mask[:, index_x, index_y]
        
        image1 = image[:, index_x-n : index_x+n+1, index_y-m : index_y+m+1]
        image2 = image[:, index_x-n2 : index_x+n2+1, index_y-m2 : index_y+m2+1]
        image3 = image[:, index_x-n3 : index_x+n3+1, index_y-m3 : index_y+m3+1]
        
        return image3, image2, image1, mask, image_filename, (index_x, index_y)
