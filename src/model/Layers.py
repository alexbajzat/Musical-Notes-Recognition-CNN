from src.utils.processing import *
from src.model.Activations import ReLUActivation


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

    def getWeights(self):
        return self.__weights


class ConvLayer(object):
    '''
    params is a map containing convs configuration:
        receptiveFieldSize is the size of the feature
        stride is the 'step' of the conv
        zeroPadding is the size of extending around the model
        features is an array of feature matrices
    '''

    def __init__(self, params, hyperParams):
        self.__receptiveFieldSize = params['receptiveFieldSize']
        self.__stride = params['stride']
        self.__zeroPadding = (int)((self.__receptiveFieldSize - 1) / 2)
        self.__featureNumber = params['f_number']

        size = self.__receptiveFieldSize * self.__receptiveFieldSize

        # features should be of shape (f_number X 1 X size X size) but I skipped this a bit
        # and flattened `em to 1 X size * size , further needs
        self.__features = np.random.randn(self.__featureNumber, size)

        self.__hyperparams = hyperParams

    '''
            X is an array of pixel matrices
            returns the features map
        '''

    def forward(self, X):
        XCol = im2col_indices(X, self.__receptiveFieldSize, self.__receptiveFieldSize, self.__zeroPadding,
                              self.__stride)
        weighted = np.dot(self.__features, XCol)

        # reshape is done assuming that after the conv, the feature maps keep the dims of the input
        # using magic padding
        reshaped = weighted.reshape((X.shape[0], self.__featureNumber, X.shape[2], X.shape[3]))

        self.__cache = X, XCol
        return reshaped

    '''
        gradients is of size (input_n X filter_n X filter_h X filter_w)  
    '''

    def backprop(self, gradients):
        X, XCol = self.__cache

        # reshape gradients for compatibilty: (filter_N X filter_h X filter_W X input_n) and reshape to (filter_N X filter_h * filter_w & input_n)
        gradientsReshaped = gradients.transpose(1, 2, 3, 0).reshape(self.__featureNumber, -1)

        # calculate gradients on feature
        dFeatures = np.dot(gradientsReshaped, np.transpose(XCol))
        self.__features += - self.__hyperparams.featureStepSize * dFeatures

        # calculate gradients on input
        dXCol = np.dot(np.transpose(self.__features), gradientsReshaped)
        dX = col2im_indices(dXCol, X.shape, self.__receptiveFieldSize, self.__receptiveFieldSize, self.__zeroPadding,
                            self.__stride)

        return dX

    def getFeatures(self):
        return self.__features


'''
    layer class which just passes the input thru an activation
'''


class PoolLayer(object):
    def __init__(self, size=2, stride=2, type='MAX'):
        self.__size = size
        self.__stride = stride
        self.__type = type

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
        self.__cache = X, XCol, maxIndexes

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


class REluActivationLayer(object):
    def __init__(self):
        self.__activation = ReLUActivation()

    def forward(self, X):
        return self.__activation.apply(X)

    def backward(self, X, gradients):
        return self.__activation.derivative(X, gradients)


class FlattenLayer(object):

    def forward(self, X):
        self.__cache = X.shape
        return X.reshape(X.shape[0], -1)

    def backprop(self, gradients):
        return gradients.reshape(self.__cache)
