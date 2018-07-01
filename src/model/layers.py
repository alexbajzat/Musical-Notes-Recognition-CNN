from src.model.activations import NonActivation
from src.utils.processing import *


class HiddenLayer(object):
    '''
    inputSize is the feature size
    output is the size of next layer
    activation class
    hyperparams bundle of nn parameters and stuff
    '''

    def __init__(self, inputSize=None, outputSize=None, activation=NonActivation(), hyperparams=None, weights=None,
                 biases=None):
        self.__inputSize = inputSize
        self.__outputSize = outputSize
        self.__activation = activation
        self.__hyperparams = hyperparams
        self.__weights = np.asarray(weights)
        self.__biases = np.asarray(biases)
        # initialize weights and biases
        if (inputSize != None and outputSize != None):
            self.__weights = np.random.randn(inputSize, outputSize) * 1e-1
            self.__biases = np.zeros((1, outputSize)) + 0.01

    def forward(self, X):
        result = np.dot(X, self.__weights) + self.__biases
        apply = self.__activation.apply(result)
        self.__cache = (deepcopy(X), deepcopy(result), deepcopy(apply))
        return apply

    def backprop(self, gradients):
        X, XWeighted,  XActivated = self.__cache

        gradientsAct = self.__activation.derivative(XWeighted, gradients)

        deltaWeights = np.dot(np.transpose(X), gradientsAct)
        deltaBiases = np.sum(gradients, axis=0, keepdims=True)

        # deltaWeights += self.__weights * self.__hyperparams.regularization

        newGradient = np.dot(gradientsAct, np.transpose(self.__weights))

        self.__weights += - self.__hyperparams.stepSize * deltaWeights
        self.__biases += - self.__hyperparams.stepSize * deltaBiases

        return newGradient

    def getWeights(self):
        return self.__weights

    def getFormattedWeights(self):
        return self.__weights

    def getActivation(self):
        return self.__activation

    def getBiases(self):
        return self.__biases


class ConvLayer(object):
    '''
    params is a map containing convs configuration:
        receptiveFieldSize is the size of the feature
        stride is the 'step' of the conv
        zeroPadding is the size of extending around the model
        features is an array of feature matrices
    '''

    def __init__(self, params=None, hyperParams=None, activation=NonActivation(), inputDepth=1, filters=None, stride=1):
        if (filters != None):
            self.__receptiveFieldSize = len(filters[0])
            self.__filterNumber = len(filters)
            self.__filters = np.asarray(filters)
            self.__receptiveFieldSize = self.__filters.shape[2]
            self.__filters = self.__filters.reshape(self.__filters.shape[0], self.__filters.shape[1], -1)
            self.__stride = stride
        if (params != None):
            self.__receptiveFieldSize = params.receptive_field_size
            self.__filterNumber = params.filter_number
            self.__stride = params.stride
            size = self.__receptiveFieldSize * self.__receptiveFieldSize
            min, max = (params.filter_distribution_interval, -1 * params.filter_distribution_interval)
            # features should be of shape (f_number X 1 X size X size) but I skipped this a bit
            # and flattened `em to 1 X size * size , further needs
            self.__filters = np.random.uniform(min, max, (self.__filterNumber, inputDepth, size))

            self.__hyperparams = hyperParams

        self.__zeroPadding = (int)((self.__receptiveFieldSize - 1) / 2)
        self.__activation = activation

    '''
            X is an array of pixel matrices
            returns the features map
        '''

    def forward(self, X):
        # XReshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[3])
        XCol = im2col_indices(X, self.__receptiveFieldSize, self.__receptiveFieldSize, self.__zeroPadding,
                              self.__stride)

        outHeight = int((X.shape[2] - self.__receptiveFieldSize + 2 * self.__zeroPadding) / self.__stride + 1)
        outWidth = int((X.shape[3] - self.__receptiveFieldSize + 2 * self.__zeroPadding) / self.__stride + 1)

        filters = self.__filters.reshape(self.__filters.shape[0], -1)
        weighted = np.dot(filters, XCol)

        reshaped = weighted.reshape((self.__filterNumber, outHeight, outWidth, X.shape[0]))

        transpose = reshaped.transpose(3, 0, 1, 2)

        apply = self.__activation.apply(transpose)
        self.__cache = deepcopy(X), deepcopy(apply), deepcopy(XCol), deepcopy(transpose)

        return apply

    '''
        gradients is of size (input_n X filter_n X filter_h X filter_w)  
    '''

    def backprop(self, gradients):
        X, XAct, XCol, XWeighted = self.__cache

        activationBack = self.__activation.derivative(XWeighted, gradients)

        # reshape gradients for compatibilty: (filter_N X filter_h X filter_W X input_n)
        # and reshape to (filter_N X filter_h * filter_w * input_n)
        gradientsReshaped = activationBack.transpose(1, 2, 3, 0).reshape(self.__filterNumber, -1)

        # calculate gradients on feature
        dFeatures = np.dot(gradientsReshaped, np.transpose(XCol))
        dFeatures = np.reshape(dFeatures, self.__filters.shape)


        # calculate gradients on input
        dXCol = np.dot(np.transpose(self.__filters), gradientsReshaped)
        dX = col2im_indices(dXCol, X.shape, self.__receptiveFieldSize, self.__receptiveFieldSize, self.__zeroPadding,
                            self.__stride)

        #update values
        self.__filters += -self.__hyperparams.featureStepSize * dFeatures

        return dX

    def getFilters(self):
        return self.__filters, self.__filterNumber, self.__receptiveFieldSize

    def getWeights(self):
        return self.__filters

    def getBiases(self):
        return np.empty((1, 0))

    def getFormattedWeights(self):
        return self.__filters.reshape(self.__filters.shape[0], self.__filters.shape[1], self.__receptiveFieldSize,
                                      self.__receptiveFieldSize)

    def getActivation(self):
        return self.__activation

    def getConvParams(self):
        return {"stride": self.__stride}


class PoolLayer(object):
    def __init__(self, size=2, stride=2, type='MAX'):
        self.__size = size
        self.__stride = stride
        self.__type = type
        self.__activation = NonActivation()

    def forward(self, X):
        # reshape X (merging the feature size and input size) for the im2col
        XReshaped = X.reshape(X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[2])

        # the size of input at the end of pooling operation
        outHeight = (int)(X.shape[2] / 2)
        outWidth = (int)(X.shape[3] / 2)

        XCol = im2col_indices(XReshaped, self.__size, self.__size, padding=0, stride=self.__stride)

        # indexes of max element of every patch
        maxIndexes = np.argmax(XCol, axis=0)

        # max elements from every patch
        resized = XCol[maxIndexes, range(maxIndexes.size)]

        # give the shape
        reshaped = resized.reshape(outHeight, outWidth, X.shape[0], X.shape[1])

        # save X and XCol for the backward pass
        self.__cache = deepcopy(X), deepcopy(XCol), deepcopy(maxIndexes)

        # return to normal forms
        return reshaped.transpose(2, 3, 0, 1)

    def backprop(self, gradient):
        X, XCol, maxIndexes = self.__cache

        dXCol = np.zeros_like(XCol)

        # i_number x f_number x f_height x f_width => i_number x f_number x f_height x f_width
        # , then flattened to i_number * f_number * f_height * f_width
        # Transpose step is necessary to get the correct arrangement
        gradientFlattened = gradient.transpose(2, 3, 0, 1).ravel()

        # Fill the maximum index of each column with the gradient
        # Essentially putting each of the 9800 grads
        # to one of the 4 row in 9800 locations, one at each column
        dXCol[maxIndexes, range(maxIndexes.size)] = gradientFlattened

        dX = col2im_indices(dXCol, (X.shape[0] * X.shape[1], 1, X.shape[2], X.shape[2]), self.__size, self.__size,
                            padding=0, stride=self.__stride)
        return dX.reshape(X.shape)

    def getWeights(self):
        return np.empty(0)

    def getFormattedWeights(self):
        return np.empty(0)

    def getActivation(self):
        return self.__activation


class FlattenLayer(object):
    def __init__(self):
        self.__activation = NonActivation()

    def forward(self, X):
        self.__cache = X.shape
        return X.reshape(X.shape[0], -1)

    def backprop(self, gradients):
        return gradients.reshape(self.__cache)

    def getWeights(self):
        return np.empty(0)

    def getFormattedWeights(self):
        return np.empty(0)

    def getActivation(self):
        return self.__activation


'''
    this does not affect the derivative, only resizes data to be processed
'''


class TestingLayer(object):
    def __init__(self, inputSize, outputSize):
        self.__inputSize = inputSize
        self.__outputSize = outputSize
        self.__activation = NonActivation()
        # initialize weights and biases
        # todo weights should be initialized using square root / input size
        self.__weights = np.random.randn(inputSize, outputSize) * 0.01

    def forward(self, X):
        return np.dot(X, self.__weights)

    def backprop(self, gradients):
        newGradient = np.dot(gradients, np.transpose(self.__weights))
        return newGradient

    def getWeights(self):
        return self.__weights

    def getFormattedWeights(self):
        return self.__weights

    def getActivation(self):
        return self.__activation

    def getBiases(self):
        return np.empty((1, 0))
