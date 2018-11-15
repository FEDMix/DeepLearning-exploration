import numpy as np
import torch
from scipy.stats import norm
import random
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

import os
import tempfile
import datetime
import time
import sys
#sys.setdefaultencoding("utf-8")

import pydicom
from pydicom.dataset import Dataset, FileDataset
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    np_image = image.detach().cpu().numpy().astype(np.float32)
    print np.mean(np_image)
    if np_image.shape[0] > 1:
        plt.imshow(np.transpose(np_image, (1, 2, 0)), interpolation='nearest', cmap='gray', norm=NoNorm())
    else:
        plt.imshow(np_image[0], interpolation='nearest', cmap='gray', norm=NoNorm())
    plt.title(title)
    plt.show()


def show_grid(images, title = ''):
    np_images = images.detach().cpu().numpy().astype(np.float32)
    print np.mean(np_images)
    if np_images.shape[0] > 1:
        plt.imshow(np.transpose(np_images, (1, 2, 0)), interpolation='nearest', cmap='gray', norm=NoNorm())
    else:
        plt.imshow(np_images[0], interpolation='nearest', cmap='gray', norm=NoNorm())
    plt.title(title)
    plt.show()

def save_dicom_auto(image, patient, volume):
    mrSegmentationForExportShItemID = patient*1000 + volume
    mrVolumeForExportShItemID = patient*1000 + volume
    
    import DicomRtImportExportPlugin
    exporter = DicomRtImportExportPlugin.DicomRtImportExportPluginClass()
    exportables = []
    mrSegmentationExportable = exporter.examineForExport(mrSegmentationForExportShItemID)
    #mrVolumeExportable = exporter.examineForExport(mrVolumeForExportShItemID)
    #exportables.extend(mrVolumeExportable)
    exportables.extend(mrSegmentationExportable)
    for exp in exportables:
        exp.directory = 'output/'
    exporter.export(exportables)
    
def save_dicom(image, filename, patient, volume):
    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
    image = image[0]
    #if np.max(image.flatten()) > 0:
    #    image /= float(np.max(image.flatten()))
    image = np.random.uniform(0,1,image.shape)
    image *= 16.0
    filename = 'output/' + filename + '.dcm'
    
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = 'Secondary Capture Image Storage'
    file_meta.MediaStorageSOPInstanceUID = '1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780'
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'
    ds = FileDataset(filename, {},file_meta = file_meta,preamble=b'\0'*128)
    ds.Modality = 'MRI'
    
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S')  # long format with micro seconds
    ds.ContentTime = timeStr
    print ds.ContentTime
    
    ds.PatientName = str(patient)
    ds.PatientID = str(patient)
    ds.SOPInstanceUID =    str(volume)
    ds.StudyInstanceUID =  '1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093'
    ds.SeriesInstanceUID = '1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649'
    ds.SecondaryCaptureDeviceManufctur = 'Python2'

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = b'\\x00\\x00'
    ds.LargestImagePixelValue = b'\\xff\\xff'
    ds.Columns = image.shape[0]
    ds.Rows = image.shape[1]
    if image.dtype != np.uint16:
        image = image.astype(np.uint16)
    
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    ds.PixelData = image.tostring()
    print len(image.tostring())

    ds.save_as(filename)
    return

def save_prediction(x_test, y_test, prediction, filename):
    if len(x_test.shape) == 3:
        x_test = np.expand_dims(x_test, axis=0)
        y_test = np.expand_dims(y_test, axis=0)
        prediction = np.expand_dims(prediction, axis=0)
    
   
    x_test = np.swapaxes(x_test,1,2)
    x_test = np.swapaxes(x_test,2,3)
    y_test = np.swapaxes(y_test,1,2)
    y_test = np.swapaxes(y_test,2,3)
    prediction = np.swapaxes(prediction,1,2)
    prediction = np.swapaxes(prediction,2,3)
    if x_test.shape[3] == 1:
        x_test = x_test[:, :, :, 0]
        y_test = y_test[:, :, :, 0]
        prediction = prediction[:, :, :, 0]
        
    
    #print np.min(x_test), np.max(x_test)
    #print np.min(y_test), np.max(y_test)
    #print np.min(prediction), np.max(prediction)
    
    #print x_test.shape, y_test.shape, prediction.shape
    test_size = x_test.shape[0]
    fig, ax = plt.subplots(nrows = test_size, ncols = 3, figsize=(22,8), sharey=False, sharex=False)
    
    #divider = make_axes_locatable(ax)
    #ccax = divider.append_axes("right", size="5%", pad=0.05)
    
    ax = np.atleast_2d(ax)

    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        #lt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i])
        #plt.colorbar(cax, ax=ax[i,1])
        cax = ax[i, 2].imshow(prediction[i])
        #plt.colorbar(cax, cax=ccax)
        if i==0:
            ax[i, 0].set_title("input scan")
            ax[i, 1].set_title("true segmentation")
            ax[i, 2].set_title("prediction")
    fig.tight_layout()
    
    fig.savefig(filename)
    plt.close()
    #plt.show()