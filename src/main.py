import random

import numpy as np

from src.NeuralModel import Model
from src.data.Setup import initDataset, Constants
from src.data.constants import LayerType
from src.data.mnistdata import initMNISTDataset
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.model.LayersBuilder import LayersBuilder
from src.utils.processing import augmentateDataset


def doTheStuff(data):
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
    trainingUpperBound = datasetSize - BATCH_SIZE
    trainingDataset = datasetValues[0:trainingUpperBound], datasetLabels[0:trainingUpperBound]
    validatingDataset = datasetValues[trainingUpperBound:], datasetLabels[trainingUpperBound:]

    # construct layers
    layersBuilder = LayersBuilder()
    layersBuilder.addLayer((LayerType.CONV, {'receptive_field_size': 3, 'activation': ReLUActivation()
        , 'stride': 1, 'zero_padding': 0, 'filter_number': 15
        , 'filter_distribution_interval': (-1e-4, 1e-4)}))
    layersBuilder.addLayer((LayerType.POOLING, {}))
    # layersBuilder.addLayer((LayerType.CONV, {'receptive_field_size': 3, 'activation': ReLUActivation()
    #     , 'stride': 1, 'zero_padding': 0, 'filter_number': 10
    #     , 'filter_distribution_interval': (-1e-4, 1e-4)}))
    # layersBuilder.addLayer((LayerType.POOLING, {}))
    layersBuilder.addLayer((LayerType.FLAT, {}))
    layersBuilder.addLayer((LayerType.HIDDEN, {'activation' : ReLUActivation()}))
    layersBuilder.addLayer((LayerType.HIDDEN, {'activation' : NonActivation()}))


    # training params
    STEP_SIZE = 1e-3
    FILTER_STEP_SIZE = 1e-3
    REG = 1e-3
    hyperParams = HyperParams(STEP_SIZE, FILTER_STEP_SIZE, REG)

    # build layers and model
    layers = layersBuilder.build(hyperParams, inputDims, 100, 7)
    model = Model(layers, SoftMax(hyperParams), BATCH_SIZE, iterations=70)


    # model getting trained
    model.train(trainingDataset, validatingDataset)
    return model


def trainWithMnist():
    data = initMNISTDataset()
    doTheStuff(data)


def train():
    data = initDataset()
    doTheStuff(data)

def augmentate():
    augmentateDataset()

train()
