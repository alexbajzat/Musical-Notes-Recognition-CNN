import numpy as np
from src.Setup import initDataset
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Layer import HiddenLayer


class Model(object):
    def __init__(self, inputSize, classifier, hyperParams):
        self.__firstHiddenLayer = HiddenLayer(inputSize, 50, NonActivation(), hyperParams)
        self.__secondHiddenLayer = HiddenLayer(50, 10, ReLUActivation(), hyperParams)
        self.__classifier = classifier
        self.__hyperParams = hyperParams

    def train(self, dataset):
        data = dataset[0]
        labels = dataset[1]
        for iteration in range(100):
            print("iteration: " + iteration, '\n')

            f = self.__firstHiddenLayer.forward(data)
            s = self.__secondHiddenLayer.forward(f)
            scores = self.__classifier.compute(s, labels)

            sGrads=  self.__secondHiddenLayer.backpropagate(s, scores)
            self.__firstHiddenLayer.backpropagate(f, sGrads)


    def validate(self, dataset):
        data = dataset[0]
        labels = dataset[1]
        f = self.__firstHiddenLayer.forward(data)
        s = self.__secondHiddenLayer.forward(f)
        scores = self.__classifier.compute(s, labels)
        predictedClasses = np.argmax(scores, axis=1)
        print('training accuracy:', (np.mean(predictedClasses == labels)))

def doTheStuff():
    data = initDataset()
    dataset = [e.getData() for e in data], [e.getLabel() for e in data]
    print(dataset)

doTheStuff()