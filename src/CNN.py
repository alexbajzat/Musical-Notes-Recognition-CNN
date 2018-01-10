import numpy as np
from PIL import Image
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Layers import HiddenLayer, ConvLayer, PoolLayer, FlattenLayer, REluActivationLayer


class Model(object):
    def __init__(self, inputSize, nOfLabels, classifier, hyperParams, convParams, batchSize):
        self.__firstConvLayer = ConvLayer(params=convParams, hyperParams=hyperParams)
        self.__fPoolingLayer = PoolLayer()
        self.__sPoolingLayer = PoolLayer()
        self.__reluLayer = REluActivationLayer()
        self.__flattenLayer = FlattenLayer()
        self.__entrySize = 14
        self.__entryChannelSize = 1
        self.__nOfLabels = nOfLabels
        # todo remove hardcoded stuff
        self.__firstHiddenLayer = HiddenLayer(32 * 32 * 10, 50, ReLUActivation(), hyperParams)
        self.__secondHiddenLayer = HiddenLayer(50, nOfLabels, NonActivation(), hyperParams)
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

                # conv stuff
                fConvForward = self.__firstConvLayer.forward(batchedData)
                fRelu = self.__reluLayer.forward(fConvForward)

                # some visible proof
                if (iteration > 90):
                    self.__saveImageFeatured(fConvForward[0], 'conv')
                #
                fPoolForward = self.__fPoolingLayer.forward(fRelu)


                flatten = self.__flattenLayer.forward(fPoolForward)

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
                fPoolBack = self.__fPoolingLayer.backprop(unflatten)
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

        # save features as pngs
        for feature in self.__firstConvLayer.getFeatures():
            parsed = feature.reshape(3, 3)
            Image.fromarray(parsed, 'L').resize((100, 100)).save('../features/' + str(id(parsed)) + '.png')

        self.__saveWeights()

    def __saveImageFeatured(self, featured, opType):
        for img in featured:
            Image.fromarray(img, 'L').resize((100, 100)).save('../features/' + opType + "-" + str(id(img)) + '.png')

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
