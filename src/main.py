import numpy as np
import random
from src.data.Setup import initDataset, Constants
from src.data.mnistdata import initMNISTDataset
from src.model.Activations import ReLUActivation, NonActivation
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.CNN import Model
from src.model.Layers import ConvLayer, PoolLayer, REluActivationLayer, FlattenLayer, HiddenLayer
from src.data.constants import LayerType

STEP_SIZE = 1e-6
FEATURE_STEP_SIZE = 1e-1
REG = 1e-3
BATCH_SIZE = 100


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

    trainingDataset = datasetValues[0:400], datasetLabels[0:400]
    validatingDataset = datasetValues[400:], datasetLabels[400:]
    hyperParams = HyperParams(STEP_SIZE, REG, FEATURE_STEP_SIZE)

    params = {'receptiveFieldSize': 3, 'stride': 1, 'zeroPadding': None, 'f_number': 10}
    nOfLabels = 7

    layers = []
    layers.append((ConvLayer(params=params, hyperParams=hyperParams), LayerType.CONV))
    layers.append((PoolLayer(), LayerType.POOLING))
    layers.append((ConvLayer(params=params, hyperParams=hyperParams,
                             featureDepth=params['f_number']), LayerType.CONV))
    layers.append((PoolLayer(), LayerType.POOLING))
    layers.append((REluActivationLayer(), LayerType.ACTIVATION))
    layers.append((FlattenLayer(), LayerType.FLAT))
    # watch-out for the input size of the first fully net
    # /4 comes from the number of pooling layers ( they shrink 2X the data)
    layers.append((HiddenLayer(int(np.power(inputSize / 4, 2) * params['f_number']),
                               100, ReLUActivation(), hyperParams), LayerType.HIDDEN))
    layers.append((HiddenLayer(100, nOfLabels, NonActivation(), hyperParams), LayerType.HIDDEN))
    classifier = SoftMax(hyperParams)

    # model getting trained
    model = Model(layers, classifier, BATCH_SIZE)
    model.train(trainingDataset)
    model.validate(validatingDataset)
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
