import random

import numpy as np

from src.data.Setup import initDataset, Constants
from src.data.constants import LayerType
from src.data.mnistdata import initMNISTDataset
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.model.Activations import ReLUActivation, NonActivation
from src.model.Layers import HiddenLayer, ConvLayer, PoolLayer, REluActivationLayer, FlattenLayer
from src.NeuralModel import Model

STEP_SIZE = 1e-6
FEATURE_STEP_SIZE = 1e-6
REG = 1e-4
BATCH_SIZE = 32
FULLY_CONNECTED_NEURONS = 50
LABELS_NUMBER = 7


def doTheStuff(data):
    inputSize = data[0].getData().shape[1]

    # randomize data for better distribution
    random.shuffle(data)
    datasetSize = len(data)

    # initialize data
    datasetValues = np.empty((datasetSize, Constants.CHANNEL_SIZE, inputSize, inputSize), dtype=int)
    datasetLabels = np.empty((datasetSize, 1), dtype=int)
    position = 0
    for value in data:
        datasetValues[position] = value.getData()
        datasetLabels[position] = value.getLabel()
        position += 1

    # dataset - batch-size = amount of data trained
    #
    trainingUpperBound = datasetSize - BATCH_SIZE

    trainingDataset = datasetValues[0:trainingUpperBound], datasetLabels[0:trainingUpperBound]
    validatingDataset = datasetValues[trainingUpperBound:], datasetLabels[trainingUpperBound:]
    hyperParams = HyperParams(STEP_SIZE, REG, FEATURE_STEP_SIZE)

    fConvparams = {'receptiveFieldSize': 3, 'stride': 1, 'zeroPadding': None, 'f_number': 5
        , 'filter_distribution_interval': (-1, 1)}
    sConvparams = {'receptiveFieldSize': 3, 'stride': 1, 'zeroPadding': None, 'f_number': 15
        , 'filter_distribution_interval': (-1, 1)}

    # init layers
    layers = []

    # add `em

    # conv-relu-conv-relu-pool
    layers.append((ConvLayer(params=fConvparams, hyperParams=hyperParams), LayerType.CONV))
    layers.append((REluActivationLayer(), LayerType.ACTIVATION))
    layers.append(
        (ConvLayer(params=fConvparams, hyperParams=hyperParams, featureDepth=fConvparams['f_number']), LayerType.CONV))
    layers.append((REluActivationLayer(), LayerType.ACTIVATION))
    layers.append((PoolLayer(), LayerType.POOLING))

    # conv-relu-pool
    layers.append(
        (ConvLayer(params=sConvparams, hyperParams=hyperParams, featureDepth=fConvparams['f_number']), LayerType.CONV))
    layers.append((REluActivationLayer(), LayerType.ACTIVATION))
    layers.append((PoolLayer(), LayerType.POOLING))

    layers.append((FlattenLayer(), LayerType.FLAT))
    # watch-out for the input size of the first fully net
    # /4 comes from the number of pooling layers ( they shrink 2X the data)

    # shrink is done with 2^n_of_pools
    poolN = len([t for t in layers if t[1] == LayerType.POOLING])
    inputShrink = np.power(2, poolN)
    fHiddenInput = int(np.power(inputSize / inputShrink, 2) * sConvparams['f_number'])
    layers.append((HiddenLayer(fHiddenInput,
                               FULLY_CONNECTED_NEURONS, NonActivation(), hyperParams), LayerType.HIDDEN))
    layers.append((HiddenLayer(FULLY_CONNECTED_NEURONS, LABELS_NUMBER, NonActivation(), hyperParams), LayerType.HIDDEN))

    classifier = SoftMax(hyperParams)

    # model getting trained
    model = Model(layers, classifier, BATCH_SIZE, iterations=50)
    model.train(trainingDataset, validatingDataset)
    return model


def trainWithMnist():
    data = initMNISTDataset()
    doTheStuff(data)


def train():
    data = initDataset()
    doTheStuff(data)


def play():
    while (True):
        input('Press anything to predict')
        print('predicting... ')


train()
