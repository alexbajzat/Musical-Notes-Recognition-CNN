import numpy as np
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Layers import HiddenLayer, ConvLayer, PoolLayer, FlattenLayer


class Model(object):
    def __init__(self, inputSize, classifier, hyperParams, convParams):
        self.__firstConvLayer = ConvLayer(params=convParams, hyperParams=hyperParams)
        self.__poolingLayer = PoolLayer()
        self.__flattenLayer = FlattenLayer()
        self.__firstHiddenLayer = HiddenLayer(32 * 32 * 5, 100, ReLUActivation(), hyperParams)
        self.__secondHiddenLayer = HiddenLayer(100, 7, NonActivation(), hyperParams)
        self.__classifier = classifier
        self.__hyperParams = hyperParams

    def train(self, dataset):
        data = dataset[0]
        labels = dataset[1]
        for iteration in range(100):
            print("iteration: ", iteration, '\n')

            # conv stuff
            fConvForward = self.__firstConvLayer.forward(data)
            fPoolForward = self.__poolingLayer.forward(fConvForward)
            flatten = self.__flattenLayer.forward(fPoolForward)

            # Fully connected start
            f = self.__firstHiddenLayer.forward(flatten)
            s = self.__secondHiddenLayer.forward(f)

            # gradients on score
            scores = self.__classifier.compute(s, labels)

            sGrads = self.__secondHiddenLayer.backpropagate(f, scores)
            firstHiddenGrads = self.__firstHiddenLayer.backpropagate(flatten, sGrads)

            # backprop into flatten layer
            # from here we backprop to convs
            unflatten = self.__flattenLayer.backprop(firstHiddenGrads)

            fPoolBack = self.__poolingLayer.backprop(unflatten)
            self.__firstConvLayer.backprop(fPoolBack)

            # done propagating to convs

    def validate(self, dataset):
        data = dataset[0]
        labels = dataset[1]

        # conv
        fConvForward = self.__firstConvLayer.forward(data)  
        fPoolForward = self.__poolingLayer.forward(fConvForward)
        flatten = self.__flattenLayer.forward(fPoolForward)

        # Fully connected start
        f = self.__firstHiddenLayer.forward(flatten)
        s = self.__secondHiddenLayer.forward(f)
        predictedClasses = np.argmax(s, axis=1)
        print('training accuracy:', (np.mean(predictedClasses == labels)))
