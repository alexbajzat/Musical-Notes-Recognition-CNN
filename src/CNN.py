import numpy as np
from src.Setup import initDataset
from src.model.Activations import NonActivation, ReLUActivation
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.model.Layer import HiddenLayer


'''
    hyperparamenters
'''
STEP_SIZE =1e-07
REG =1e-3

class Model(object):
    def __init__(self, inputSize, classifier, hyperParams):
        self.__firstHiddenLayer = HiddenLayer(inputSize, 100, NonActivation(), hyperParams)
        self.__secondHiddenLayer = HiddenLayer(100, 2, ReLUActivation(), hyperParams)
        self.__classifier = classifier
        self.__hyperParams = hyperParams

    def train(self, dataset):
        data = dataset[0]
        labels = dataset[1]
        for iteration in range(100):
            print("iteration: " , iteration , '\n')

            f = self.__firstHiddenLayer.forward(data)
            s = self.__secondHiddenLayer.forward(f)
            scores = self.__classifier.compute(s, labels)

            sGrads=  self.__secondHiddenLayer.backpropagate(f, scores)
            self.__firstHiddenLayer.backpropagate(data, sGrads)


    def validate(self, dataset):
        data = dataset[0]
        labels = dataset[1]
        f = self.__firstHiddenLayer.forward(data)
        s = self.__secondHiddenLayer.forward(f)
        predictedClasses = np.argmax(s, axis=1)
        print('training accuracy:', (np.mean(predictedClasses == labels)))
'''
    translate the pixel matrix to vector for the input of the model, numpy is magic
'''
def flattenMatrix(data):
    return np.array(data).reshape(-1)

def doTheStuff():
    data = initDataset()
    datasetSize = len(data)

    dataset = [flattenMatrix(e.getData().getdata()) for e in data], [int(e.getLabel()) for e in data]
    print(dataset)
    inputSize = len(dataset[0][0])
    hyperParams = HyperParams(STEP_SIZE, REG)

    model = Model(inputSize, SoftMax(datasetSize, hyperParams), hyperParams)
    model.train(dataset)
    model.validate(dataset)





doTheStuff()