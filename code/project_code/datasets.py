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
            
            if np.random.uniform < 0.5:
                image, mask = uniform_rotate(image, mask, -30, 30)
            else:
                image, mask = shear(image, mask, 30)
                
            if np.random.uniform(0, 1) < 0.2:
                image, mask = rotate_90(image, mask)

            if np.random.uniform(0, 1) < 0.3:
                image, mask = hflip(image, mask)

        if image.size[0] != self.image_dim[0] or image.size[1] != self.image_dim[1]:
            image, mask = resize(image, mask, self.image_dim)

        #image = image.expand(3, -1, -1)
        
        #image1, mask1 = resize(image, mask, (64,64))
        #image2, mask2 = resize(image, mask, (128,128))
        #image3, mask3 = resize(image, mask, (256,256))
        #image1 = to_tensor(image1)
        #image2 = to_tensor(image2)
        #image3 = to_tensor(image3)
        
        image, mask = to_tensor(image), to_tensor(mask)
        return image, mask, image_filename
        #return image, image1, image2, image3, mask, image_filename
