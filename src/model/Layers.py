from src.utils.preprocessing import *


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
        self.__zeroPadding = (int)((self.__receptiveFieldSize - 1) / 2)
        self.__featureNumber = params['f_number']

        size = self.__receptiveFieldSize * self.__receptiveFieldSize
        self.__features = np.empty((1, size))
        self.__biases = np.zeros((size, 1), dtype=int)
        np.insert(self.__features, 1, np.random.randn(size))

    '''
        X is an array of pixel matrices
        returns the features map
    '''

    def forward(self, X):
        XCol = im2col(X, self.__receptiveFieldSize, self.__receptiveFieldSize, self.__zeroPadding, self.__stride)
        # TODO add biases
        weighted = np.dot(self.__features, XCol)
        reshaped = weighted.reshape((len(self.__features), X.shape[1], X.shape[2], X.shape[0]))

        self.__cache = X, XCol
        return reshaped.transpose(3, 0, 1, 2)

    '''
        gradients is of size (input_n X filter_n X filter_h X filter_w)  
    '''

    def backprop(self, gradients):
        X, XCol = self.__cache
        # reshape gradients for compatibilty: (filter_N X filter_h X filter_W X input_n) and reshape to (filter_N X filter_h * filter_w & input_n)
        gradientsReshaped = gradients.transpose(1, 2, 3, 0).reshape(self.__featureNumber, -1)

        # calculate gradients on feature
        dFeatures = np.dot(gradientsReshaped, np.transpose(XCol))
        self.__features += dFeatures

        # calculate gradients on input
        dXCol = np.dot(np.transpose(self.__features), gradientsReshaped)
        dX = col2im_indices(dXCol, X.shape, self.__receptiveFieldSize, self.__receptiveFieldSize, self.__zeroPadding,
                            self.__stride)
        return None


'''
    layer class which just passes the input thru an activation
'''


class PoolLayer(object):
    def __init__(self, size=2, stride=2, type='MAX'):
        self.__size = size
        self.__stride = stride
        self.__type = type

    def forward(self, X):
        return None


class ActivationLayer(object):

    def __init__(self, activation):
        self.__activation = activation

    def forward(self, X):
        return self.__activation.apply(X)
