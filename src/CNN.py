import numpy as np
from PIL import Image
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Layers import HiddenLayer, ConvLayer, PoolLayer, FlattenLayer


class Model(object):
    def __init__(self, inputSize, classifier, hyperParams, convParams, batchSize):
        # self.__firstConvLayer = ConvLayer(params=convParams, hyperParams=hyperParams)
        # self.__fPoolingLayer = PoolLayer()
        # self.__sPoolingLayer = PoolLayer()
        self.__flattenLayer = FlattenLayer()
        # todo remove hardcoded stuff
        self.__firstHiddenLayer = HiddenLayer(28 * 28, 50, ReLUActivation(), hyperParams)
        self.__secondHiddenLayer = HiddenLayer(50, 10, NonActivation(), hyperParams)
        self.__classifier = classifier
        self.__hyperParams = hyperParams
        self.__batchSize = batchSize

    def train(self, dataset):
        data = dataset[0]
        labels = dataset[1]
        start = 0
        endBatch = self.__batchSize
        nOfIterations = 100
        while (True):
            batchedData = data[start:endBatch]
            batchedLabels = labels[start:endBatch]
            print('batch: ', str(start), ' - ', str(endBatch))
            decreasingByIteration = nOfIterations
            initStep = self.__hyperParams.stepSize
            for iteration in range(nOfIterations):
                print("iteration: ", iteration)

                # # conv stuff
                # fConvForward = self.__firstConvLayer.forward(batchedData)
                # fPoolForward = self.__fPoolingLayer.forward(fConvForward)
                # sPoolForward = self.__sPoolingLayer.forward(fPoolForward)
                flatten = self.__flattenLayer.forward(batchedData)

                # Fully connected start
                f = self.__firstHiddenLayer.forward(flatten)
                s = self.__secondHiddenLayer.forward(f)

                # gradients on score
                scores = self.__classifier.compute(s, batchedLabels, (
                    self.__firstHiddenLayer.getWeights(), self.__secondHiddenLayer.getWeights()))

                sGrads = self.__secondHiddenLayer.backpropagate(f, scores)
                firstHiddenGrads = self.__firstHiddenLayer.backpropagate(flatten, sGrads)

                # backprop into flatten layer
                # from here we backprop to convs
                # unflatten = self.__flattenLayer.backprop(firstHiddenGrads)
                # sPoolBack = self.__sPoolingLayer.backprop(unflatten)
                # fPoolBack = self.__fPoolingLayer.backprop(sPoolBack)
                # self.__firstConvLayer.backprop(fPoolBack)
                # done propagating to convs

            # decrease step size in time, or else it gets pretty big and overflows
            # newStep = initStep / decreasingByIteration
            # print('new step size: ' , str(newStep))
            # self.__firstHiddenLayer.setStepSize(newStep)
            # self.__secondHiddenLayer.setStepSize(newStep)
            # decreasingByIteration += nOfIterations

            # next batch
            start += self.__batchSize
            endBatch = start + self.__batchSize
            if (endBatch >= len(data)):
                break

        # save features as pngs
        # for feature in self.__firstConvLayer.getFeatures():
        #     parsed = feature.reshape(3, 3)
        #     Image.fromarray(parsed, 'L').resize((100, 100)).save('../features/' + str(id(parsed)) + '.png')

    def validate(self, dataset):
        data = dataset[0]
        labels = dataset[1]

        # conv
        # fConvForward = self.__firstConvLayer.forward(data)
        # fPoolForward = self.__fPoolingLayer.forward(fConvForward)
        # sPoolForward = self.__sPoolingLayer.forward(fPoolForward)
        flatten = self.__flattenLayer.forward(data)

        # Fully connected start
        f = self.__firstHiddenLayer.forward(flatten)
        s = self.__secondHiddenLayer.forward(f)
        predictedClasses = np.argmax(s, axis=1)
        print('training accuracy:', (np.mean(predictedClasses == labels)))

    def predict(self, data):
        # conv stuff
        fConvForward = self.__firstConvLayer.forward(data)
        fPoolForward = self.__fPoolingLayer.forward(fConvForward)
        flatten = self.__flattenLayer.forward(fPoolForward)

        # Fully connected start
        f = self.__firstHiddenLayer.forward(flatten)
        s = self.__secondHiddenLayer.forward(f)
