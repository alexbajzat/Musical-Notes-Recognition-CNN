import random

import numpy as np

from src.NeuralModel import Model
from src.data.Setup import initDataset, Constants
from src.data.constants import LayerType
from src.data.mnistdata import initMNISTDataset
from src.model.Activations import ReLUActivation, NonActivation
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.model.LayersBuilder import LayersBuilder


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


    BATCH_SIZE = 32
    trainingUpperBound = datasetSize - BATCH_SIZE

    trainingDataset = datasetValues[0:trainingUpperBound], datasetLabels[0:trainingUpperBound]
    validatingDataset = datasetValues[trainingUpperBound:], datasetLabels[trainingUpperBound:]


    layersBuilder = LayersBuilder()
    layersBuilder.addLayer((LayerType.CONV, {'receptive_field_size': 3, 'activation': NonActivation(), 'stride': 1, 'zero_padding': 0
        , 'filter_number': 20, 'filter_distribution_interval': (-1e-1, 1e-1)}))
    layersBuilder.addLayer((LayerType.POOLING, {}))
    layersBuilder.addLayer((LayerType.FLAT, {}))
    layersBuilder.addLayer((LayerType.HIDDEN, {'activation' : ReLUActivation()}))
    layersBuilder.addLayer((LayerType.HIDDEN, {'activation' : ReLUActivation()}))

    STEP_SIZE = 1e-5
    FILTER_STEP_SIZE = 1e-4
    REG = 1e-3

    hyperParams = HyperParams(STEP_SIZE, REG, FILTER_STEP_SIZE)
    layers = layersBuilder.build(hyperParams, inputDims, 50, 10)
    model = Model(layers, SoftMax(hyperParams), BATCH_SIZE, iterations=50)


    # model getting trained
    model.train(trainingDataset, validatingDataset)
    return model


def trainWithMnist():
    data = initMNISTDataset()
    doTheStuff(data)


def train():
    data = initDataset()
    doTheStuff(data)


train()
