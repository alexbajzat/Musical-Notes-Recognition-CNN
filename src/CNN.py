import numpy as np
import random
from src.Setup import initDataset, loadImages
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.model.Layers import HiddenLayer, ConvLayer

'''
    hyperparamenters values
'''
STEP_SIZE = 1e-07
REG = 1e-3


class Model(object):
    def __init__(self, inputSize, classifier, hyperParams):
        self.__firstHiddenLayer = HiddenLayer(inputSize, 1000 , ReLUActivation(), hyperParams)
        self.__secondHiddenLayer = HiddenLayer(1000, 7, NonActivation(), hyperParams)
        self.__classifier = classifier
        self.__hyperParams = hyperParams

    def train(self, dataset):
        data = dataset[0]
        labels = dataset[1]
        for iteration in range(100):
            print("iteration: ", iteration, '\n')

            f = self.__firstHiddenLayer.forward(data)
            s = self.__secondHiddenLayer.forward(f)

            scores = self.__classifier.compute(s, labels)

            sGrads = self.__secondHiddenLayer.backpropagate(f, scores)
            self.__firstHiddenLayer.backpropagate(data, sGrads)

    def validate(self, dataset):
        data = dataset[0]
        labels = dataset[1]
        f = self.__firstHiddenLayer.forward(data)
        s = self.__secondHiddenLayer.forward(f)
        predictedClasses = np.argmax(s, axis=1)
        print('training accuracy:', (np.mean(predictedClasses == labels)))


def doTheStuff():
    data = initDataset()
    datasetSize = len(data)
    inputSize = len(data[0].getData())

    # randomize data for better distribution
    random.shuffle(data)

    # initialize data
    datasetValues = np.empty((datasetSize, inputSize * inputSize), dtype=int)
    datasetLabels = np.empty((datasetSize, 1), dtype=int)
    position = 0
    for value in data:
        datasetValues[position] = value.getData().reshape(-1)
        datasetLabels[position] = value.getLabel()
        position += 1

    dataset = datasetValues, datasetLabels
    hyperParams = HyperParams(STEP_SIZE, REG)

    model = Model(inputSize * inputSize, SoftMax(datasetSize, hyperParams), hyperParams)
    model.train(dataset)
    model.validate(dataset)


def predict():
    images = loadImages()
    for image in images:
        image[1].show(command='fim')


doTheStuff()
