from os import listdir
from os.path import isfile
import random

import numpy as np
from PIL import Image

from src.model.labeled_model import LabeledModel


class Constants(object):
    DATASET_ROOT = '../dataset/processed/'
    REAL_SET_ROOT = '../images'
    MODEL_ROOT ='../model-data'
    PROCESSED_RESIZE = 64, 64
    CHANNEL_SIZE = 1


''' initializing dataset from root folder
    the label is given by the directory name
    root and extension of data must be configured before initialize
'''


def initDataset():
    labeledData = []
    for folder in listdir(Constants.DATASET_ROOT):
        addedAugmented = False
        for filename in listdir(Constants.DATASET_ROOT + folder):
            if not addedAugmented:
                for filename in listdir(Constants.DATASET_ROOT + folder + '/output'):
                    url = Constants.DATASET_ROOT + folder + '/output' + "/" + filename
                    doFileParsing(url, labeledData, folder)
                addedAugmented = True
            else:
                url = Constants.DATASET_ROOT + folder + "/" + filename
                doFileParsing(url, labeledData, folder)

    return labeledData


def doFileParsing(url, dataset, identifier):
    if (isfile(url)):
        print('loading file: ' + url)
        img = Image.open(url).convert('L')
        img = img.resize(Constants.PROCESSED_RESIZE, Image.ANTIALIAS)
        img = np.asarray(img.getdata()).reshape(img.size[0], img.size[1])
        # add formal grayscale channel
        img = np.expand_dims(img, 0)
        dataset.append(LabeledModel(img, identifier))


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


def batch(data, batchSize):
    inputDims = (data[0].getData().shape[1], data[0].getData().shape[2])
    # randomize data for better distribution
    random.shuffle(data)
    datasetSize = len(data)
    # initialize data
    datasetValues = np.empty((datasetSize, Constants.CHANNEL_SIZE, inputDims[0], inputDims[1]), dtype=int)
    datasetLabels = np.empty((datasetSize, 1), dtype=int)
    position = 0
    for value in data:
        datasetValues[position] = value.getData()
        datasetLabels[position] = value.getLabel()
        position += 1

    # dataset - batch-size = amount of data trained
    # manage bath
    BATCH_SIZE = 32
    trainingUpperBound = datasetSize - batchSize
    trainingDataset = datasetValues[0:trainingUpperBound], datasetLabels[0:trainingUpperBound]
    validatingDataset = datasetValues[trainingUpperBound:], datasetLabels[trainingUpperBound:]

    return trainingDataset, validatingDataset, inputDims, batchSize
