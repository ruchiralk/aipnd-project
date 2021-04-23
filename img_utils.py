import torch
from PIL import Image
import numpy as np

''' Resize image that shortest side is 256 pixels '''
def resize_image(im, side=256):
    width, height = im.size
    if width >= height:
        ratio = width/height
        width = int(side * ratio)
        height = side
    else:
        ratio = height/width
        width = side
        height = int(side * ratio)
    im.thumbnail((width, height))
    return im


''' Center Crop the image to be  224 * 224 '''
def center_crop(im, size=(224, 224)):
    width, height = im.size
    new_width, new_height = size
    left = (width - new_width)/2
    upper = (height - new_height)/2
    lower = height - upper
    right = width - left
    return im.crop((left, upper, right, lower))

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    with Image.open(image_path) as im:
        im = resize_image(im)
        im = center_crop(im)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = np.array(im)/255
        np_image = (np_image - mean)/std
        np_image = np_image.transpose((2, 0, 1))
        return torch.tensor(np_image).float()