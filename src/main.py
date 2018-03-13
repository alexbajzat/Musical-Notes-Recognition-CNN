import random

import numpy as np

from src.data.Setup import initDataset, Constants
from src.data.constants import LayerType
from src.data.mnistdata import initMNISTDataset
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.model.Activations import ReLUActivation, NonActivation
from src.model.Layers import HiddenLayer, ConvLayer, PoolLayer, FlattenLayer, TestingLayer
from src.NeuralModel import Model

STEP_SIZE = 1e-5
FILTER_STEP_SIZE = 1e-2
REG = 1e-3
BATCH_SIZE = 32

FULLY_CONNECTED_NEURONS = 1
LABELS_NUMBER = 10
CONV_DISTRIBUTION_INTERVAL = 1e-1


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
    hyperParams = HyperParams(STEP_SIZE, REG, FILTER_STEP_SIZE)

    fConvparams = {'receptiveFieldSize': 3, 'stride': 1, 'zeroPadding': None, 'f_number': 50
        , 'filter_distribution_interval': (-CONV_DISTRIBUTION_INTERVAL, CONV_DISTRIBUTION_INTERVAL)}

    # init layers
    layers = []

    # add `em

    # conv-relu-pool
    layers.append((ConvLayer(params=fConvparams, hyperParams=hyperParams, activation=ReLUActivation()), LayerType.CONV))
    layers.append(
        (ConvLayer(params=fConvparams, hyperParams=hyperParams, activation=ReLUActivation(),
                   featureDepth=fConvparams['f_number']), LayerType.CONV))


    layers.append((PoolLayer(), LayerType.POOLING))
    layers.append((FlattenLayer(), LayerType.FLAT))

    # watch-out for the input size of the first fully net
    # /4 comes from the number of pooling layers ( they shrink 2X the data)

    # shrink is done with 2^n_of_pools
    poolN = len([t for t in layers if t[1] == LayerType.POOLING])
    inputShrink = np.power(2, poolN)
    fHiddenInput = int(np.power(inputSize / inputShrink, 2) * fConvparams['f_number'])
    # fHiddenInput = int(inputSize * inputSize * 1)
    # layers.append((HiddenLayer(fHiddenInput,
    #                            FULLY_CONNECTED_NEURONS, NonActivation(), hyperParams), LayerType.HIDDEN))
    # layers.append((HiddenLayer(fHiddenInput, LABELS_NUMBER, NonActivation(), hyperParams), LayerType.HIDDEN))
    layers.append((TestingLayer(fHiddenInput, LABELS_NUMBER), LayerType.HIDDEN))
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

trainWithMnist()
