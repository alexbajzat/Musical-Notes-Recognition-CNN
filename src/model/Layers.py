from src.utils.im2col import *


class HiddenLayer(object):
    '''
    inputSize is the feature size
    output is the size of next layer
    activation class
    hyperparams bundle of nn parameters and stuff
    '''

    def __init__(self, inputSize, outputSize, activation, hyperparams):
        self.__inputSize = inputSize
        self.__outputSize = outputSize
        self.__activation = activation
        self.__hyperparams = hyperparams

        # initialize weights and biases
        # todo weights should be initialized using square root / input size
        self.__weights = 0.01 * np.random.randn(inputSize, outputSize)
        self.__biases = np.zeros((1, outputSize))

    def forward(self, X):
        result = np.dot(X, self.__weights) + self.__biases
        return self.__activation.apply(result)

    def backpropagate(self, X, gradients):
        deltaWeights = np.dot(np.transpose(X), gradients)
        deltaBiases = np.sum(gradients, axis=0, keepdims=True)

        deltaWeights += self.__weights * self.__hyperparams.regularization

        self.__weights += - self.__hyperparams.stepSize * deltaWeights
        self.__biases += - self.__hyperparams.stepSize * deltaBiases

        newGradient = np.dot(gradients, np.transpose(self.__weights))
        return self.__activation.derivative(X, newGradient)


class ConvLayer(object):
    '''
    params is a map containing convs configuration:
        receptiveFieldSize is the size of the feature
        stride is the 'step' of the conv
        zeroPadding is the size of extending around the model
        features is an array of feature matrices
    '''
    def __init__(self, params, features):
        self.__receptiveFieldSize = params['receptiveFieldSize']
        self.__stride = params['stride']
        self.__zeroPadding = params['zeroPadding']
        if (self.__zeroPadding == None):
            self.__zeroPadding = (int) ((self.__receptiveFieldSize - 1) / 2)



    '''
        in this case X is an array of pixel matrices
    '''
    def forward(self, X):
        XCol = im2col(X, self.__receptiveFieldSize, self.__receptiveFieldSize, self.__zeroPadding, self.__stride)
        return XCol

