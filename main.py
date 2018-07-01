import os

from src.data.constants import LayerType
from src.data.setup import initDataset, batch, Constants
from src.model.classifiers import SoftMax
from src.model.hyper_params import HyperParams
from src.model.layers_builder import LayersBuilder
from src.neural_model import Model
from src.utils.processing import parseJSON

MANDATORY_FOLDERS = ['dataset', 'requests']
GENERABLE_FOLDES = ['history', 'model-data', 'features']


def constructModel(requestJSON, data):
    request = parseJSON(requestJSON)
    layersBuilder = LayersBuilder()
    trainingDataset, validatingDataset, inputDims, batchSize = batch(data, request.model.training_params.batch_size)

    hyperParams = HyperParams(request.model.training_params.step_size, request.model.training_params.filter_step_size,
                              request.model.training_params.regularization)
    iterations = request.model.training_params.iterations
    number_of_classes = request.model.training_params.number_of_classes

    for layer in request.model.layers:
        layersBuilder.addLayer((LayerType[layer.layer_type], layer.params))

    # build layers and model
    constructed = layersBuilder.build(hyperParams, inputDims, request.model.training_params.hidden_layer_size, number_of_classes)
    return Model(constructed, SoftMax(hyperParams), batchSize, iterations=iterations,
                 modelConfig=requestJSON), trainingDataset, validatingDataset


def doTheStuff(data, requestJSON):
    model, trainingData, validationData = constructModel(requestJSON, data)

    print("\n --- START TRAINING --- \n")
    # model getting trained
    model.train(trainingData, validationData)
    return model


def train():
    displayAvailableRequests()
    req = input('\nRequest name: ')
    file = open(Constants.REQUESTS_ROOT + '/' + req)

    data = initDataset()

    doTheStuff(data, file.read())


def displayAvailableRequests():
    print('Requests: ')
    for filename in os.listdir(Constants.REQUESTS_ROOT):
        print(filename, '\n')

def checkSystem():
    print('Checking system...')
    for folder in MANDATORY_FOLDERS:
        if (not os.path.exists(Constants.ROOT + '/' + folder)):
            raise Exception('Please create folder structure: ' + folder)

    for folder in GENERABLE_FOLDES:
        url = Constants.ROOT + '/' + folder
        if (not os.path.exists(url)):
            os.mkdir(url)
    print('System check done successfully')



checkSystem()
train()
