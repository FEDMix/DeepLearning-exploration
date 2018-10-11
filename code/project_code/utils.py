import numpy as np
import torch
from scipy.stats import norm
import random
import matplotlib.pyplot as plt


def set_random_seeds(gpu=False):
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    if gpu:
        torch.cuda.manual_seed(1)


def random_central_crop_by_scale(image, mask, scale=0.9):

    new_size = int(image.shape[0] * scale)

    centre_x = image.shape[0] // 2
    centre_y = image.shape[1] // 2

    normal_sampling = np.random.normal(0, 20, 2)
    offset_x, offset_y = int(normal_sampling[0]), int(normal_sampling[1])
    offset_x, offset_y = -new_size//2 + \
        int(normal_sampling[0]), -new_size//2 + int(normal_sampling[1])
    offset_x = np.clip(offset_x, -centre_x, centre_x - new_size)
    offset_y = np.clip(offset_y, -centre_y, centre_y - new_size)

    image_cropped = image[centre_x + offset_x: centre_x + offset_x +
                          new_size, centre_y + offset_y: centre_y + offset_y + new_size]
    mask_cropped = mask[centre_x + offset_x: centre_x + offset_x +
                        new_size, centre_y + offset_y: centre_y + offset_y + new_size]
    return image_cropped, mask_cropped


def random_crop_by_dim(image, mask, new_size=(128, 128)):

    if image.shape[0] - new_size[0] > 0:
        start_x = np.random.randint(0, image.shape[0] - new_size[0])
    if image.shape[1] - new_size[1] > 0:
        start_y = np.random.randint(0, image.shape[1] - new_size[1])

    image_cropped = image[start_x: start_x +
                          new_size[0], start_y: start_y + new_size[0]]
    mask_cropped = mask[start_x: start_x +
                        new_size[1], start_y: start_y + new_size[1]]

    return image_cropped, mask_cropped


def show_image(image, title = ''):
    np_image = image.detach().cpu().numpy()
    if np_image.shape[0] > 1:
        plt.imshow(np.transpose(np_image, (1, 2, 0)), interpolation='nearest')
    else:
        plt.imshow(np_image[0], interpolation='nearest')
    plt.title(title)
    plt.show()


def show_grid(images, title = ''):
    np_images = images.detach().cpu().numpy()
    if np_images.shape[0] > 1:
        plt.imshow(np.transpose(np_images, (1, 2, 0)), interpolation='nearest')
    else:
        plt.imshow(np_images[0], interpolation='nearest')
    plt.title(title)
    plt.show()
