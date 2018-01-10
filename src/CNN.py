import numpy as np
from PIL import Image
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Classifiers import SoftMax
from src.model.Layers import HiddenLayer, ConvLayer, PoolLayer, FlattenLayer, REluActivationLayer
from src.utils.processing import saveFeatureDepth


class Model(object):

    def __init__(self, inputSize, nOfLabels, hyperParams, convParams, batchSize):
        self.__firstConvLayer = ConvLayer(params=convParams, hyperParams=hyperParams)
        self.__fPoolingLayer = PoolLayer()
        self.__secondConvLayer = ConvLayer(params=convParams, hyperParams=hyperParams,
                                           featureDepth=convParams['f_number'])
        self.__sPoolingLayer = PoolLayer()
        self.__reluLayer = REluActivationLayer()
        self.__flattenLayer = FlattenLayer()
        self.__nOfLabels = nOfLabels

        # watch-out for the input size of the first fully net
        # /4 comes from the number of pooling layers ( they shrink 2X the data)
        self.__firstHiddenLayer = HiddenLayer(int(np.power(np.sqrt(inputSize / 4), 2) * convParams['f_number'] / 4),
                                              50, ReLUActivation(), hyperParams)
        self.__secondHiddenLayer = HiddenLayer(50, nOfLabels, NonActivation(), hyperParams)

        self.__classifier = SoftMax(hyperParams)

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

                # conv stuff
                fConvForward = self.__firstConvLayer.forward(batchedData)

                # some visible proof
                if (iteration > 2):
                    saveFeatureDepth(fConvForward[0], 'conv1')
                fRelu = self.__reluLayer.forward(fConvForward)
                fPoolForward = self.__fPoolingLayer.forward(fRelu)
                sConvForward = self.__secondConvLayer.forward(fPoolForward)

                # some visible proof
                if (iteration > 2):
                    saveFeatureDepth(sConvForward[0], 'conv2')
                sPoolForward = self.__sPoolingLayer.forward(sConvForward)

                flatten = self.__flattenLayer.forward(sPoolForward)

                # Fully connected start
                f = self.__firstHiddenLayer.forward(flatten)
                s = self.__secondHiddenLayer.forward(f)
                # gradients on score
                scores = self.__classifier.compute(s, batchedLabels, (
                    self.__firstHiddenLayer.getWeights(), self.__secondHiddenLayer.getWeights()))

                # backprop in fully connected
                sGrads = self.__secondHiddenLayer.backpropagate(f, scores)
                firstHiddenGrads = self.__firstHiddenLayer.backpropagate(flatten, sGrads)
                # backprop into flatten layer
                # from here we backprop to convs
                unflatten = self.__flattenLayer.backprop(firstHiddenGrads)
                sPoolBack = self.__sPoolingLayer.backprop(unflatten)
                sConvBack = self.__secondConvLayer.backprop(sPoolBack)
                fPoolBack = self.__fPoolingLayer.backprop(sConvBack)
                fReluBack = self.__reluLayer.backward(fConvForward, fPoolBack)
                self.__firstConvLayer.backprop(fReluBack)
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

    def __saveFeatures(self):
        for feature in self.__firstConvLayer.getFeatures():
            parsed = feature.reshape(3, 3)
            Image.fromarray(parsed, 'L').resize((100, 100)).save('../features/' + str(id(parsed)) + '.png')

    def validate(self, dataset):
        data = dataset[0]
        labels = dataset[1]

        # conv
        fConvForward = self.__firstConvLayer.forward(data)
        fReluForward = self.__reluLayer.forward(fConvForward)
        fPoolForward = self.__fPoolingLayer.forward(fReluForward)
        flatten = self.__flattenLayer.forward(fPoolForward)

        # Fully connected start
        f = self.__firstHiddenLayer.forward(flatten)
        s = self.__secondHiddenLayer.forward(f)
        predictedClasses = np.argmax(s, axis=1)
        print('training accuracy:', (np.mean(predictedClasses == np.transpose(labels))))

    def __saveWeights(self):
        fWeights = self.__firstHiddenLayer.getWeights()
        np.savetxt(X=fWeights, fname='../model-data/f-hidden-layer.txt',
                   header=str(fWeights.shape[0]) + ',' + str(fWeights.shape[1]) + '\n')

        sWeights = self.__secondHiddenLayer.getWeights()
        np.savetxt(X=fWeights, fname='../model-data/s-hidden-layer.txt',
                   header=str(sWeights.shape[0]) + ',' + str(sWeights.shape[1]) + '\n')

    def predict(self, data):
        # conv stuff
        fConvForward = self.__firstConvLayer.forward(data)
        fPoolForward = self.__fPoolingLayer.forward(fConvForward)
        flatten = self.__flattenLayer.forward(fPoolForward)

        # Fully connected start
        f = self.__firstHiddenLayer.forward(flatten)
        s = self.__secondHiddenLayer.forward(f)
