from PIL import Image
from src.model.LabeledModel import LabeledModel
from os import listdir
from os.path import isfile
import numpy as np


class Constants(object):
    DATASET_ROOT = '../dataset/processed/'
    REAL_SET_ROOT = '../images'
    PROCESSED_RESIZE = 64, 64
    CHANNEL_SIZE = 1


''' initializing dataset from root folder
    the label is given by the directory name
    root and extension of data must be configured before initialize
'''


def initDataset():
    labeledData = []
    for folder in listdir(Constants.DATASET_ROOT):
        for filename in listdir(Constants.DATASET_ROOT + folder):
            url = Constants.DATASET_ROOT + folder + "/" + filename
            print(url)
            if (isfile(url)):
                print('loading file: ' + url)
                img = Image.open(url).convert('L')
                img = img.resize(Constants.PROCESSED_RESIZE, Image.ANTIALIAS)
                img = np.asarray(img.getdata()).reshape(img.size[0], img.size[1])
                # add formal grayscale channel
                img = np.expand_dims(img, 0)
                labeledData.append(LabeledModel(img, folder))
    return labeledData


def loadImages():
    images = []
    for filename in listdir(Constants.DATASET_ROOT):
        url = Constants.DATASET_ROOT + "/" + filename
        print(url)
        if (isfile(url)):
            print('loading file: ' + url)
            img = Image.open(url)
            img = rgb2gray(img)
            img = img.resize(Constants.PROCESSED_RESIZE, Image.ANTIALIAS)
            parsed = np.asarray(img.getdata()).reshape(img.size[0], img.size[1])
            images.append(parsed, img)
    return images


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
