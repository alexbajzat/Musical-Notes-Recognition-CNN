import numpy as np
from PIL import Image

from src.data.constants import LayerType
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Classifiers import SoftMax
from src.model.Layers import HiddenLayer, ConvLayer, PoolLayer, FlattenLayer, REluActivationLayer
from src.utils.processing import exportPNGs


class Model(object):

    def __init__(self, layers, classifier, batchSize):
        self.__batchSize = batchSize
        self.__classifier = classifier
        self.__layers = layers

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

            for iteration in range(nOfIterations):
                print("iteration: ", iteration)
                data = batchedData
                weights = []

                # forward propagation
                for layer, type in self.__layers:
                    data = layer.forward(data)

                    # todo maybe add convs weights to regularization ?
                    if (type == LayerType.HIDDEN):
                        weights.append(layer.getWeights())

                    if (iteration > (90 / 100 * nOfIterations) and type == LayerType.CONV):
                        exportPNGs(data[0], str(type))
                # scores
                back = self.__classifier.compute(data, batchedLabels, weights)

                # backpropagation
                for layer, type in reversed(self.__layers):
                    back = layer.backprop(back)

            # next batch
            start += self.__batchSize
            endBatch = start + self.__batchSize
            if (endBatch >= len(data)):
                break

    def __saveFeatures(self, convLayer):
        filters, number, size, depth = convLayer.getFilters()
        parsed = filters.reshape(number * depth, size, size)
        exportPNGs(parsed, 'filter-conv1')

    def validate(self, dataset):
        data = dataset[0]
        labels = dataset[1]

        # forward propagation
        for layer, type in self.__layers:
            data = layer.forward(data)

        # scores
        predictedClasses = np.argmax(data, axis=1)
        print('training accuracy:', (np.mean(predictedClasses == np.transpose(labels))))

    def __saveWeights(self, layers):
        # fWeights = self.__firstHiddenLayer.getWeights()
        # np.savetxt(X=fWeights, fname='../model-data/f-hidden-layer.txt',
        #            header=str(fWeights.shape[0]) + ',' + str(fWeights.shape[1]) + '\n')
        #
        # sWeights = self.__secondHiddenLayer.getWeights()
        # np.savetxt(X=fWeights, fname='../model-data/s-hidden-layer.txt',
        #            header=str(sWeights.shape[0]) + ',' + str(sWeights.shape[1]) + '\n')
        pass

    def predict(self, data):

        # forward propagation
        for layer, type in self.__layers:
            data = layer.forward(data)

        # predict
        return np.argmax(data, axis=1)
