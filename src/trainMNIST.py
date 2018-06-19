from src.NeuralModel import Model
from src.data.Setup import batch
from src.data.constants import LayerType
from src.data.mnistdata import initMNISTDataset
from src.model.Activations import ReLUActivation, NonActivation
from src.model.Classifiers import SoftMax
from src.model.HyperParams import HyperParams
from src.model.LayersBuilder import LayersBuilder


def doTheStuff(data):
    trainingDataset, validatingDataset, inputDims, batchSize = batch(data, 32)


    # construct layers
    layersBuilder = LayersBuilder()
    layersBuilder.addLayer((LayerType.CONV, {'receptive_field_size': 3, 'activation': ReLUActivation()
        , 'stride': 1, 'zero_padding': 0, 'filter_number': 10
        , 'filter_distribution_interval': (-1e-2, 1e-2)}))
    layersBuilder.addLayer((LayerType.POOLING, {}))
    layersBuilder.addLayer((LayerType.CONV, {'receptive_field_size': 3, 'activation': ReLUActivation()
        , 'stride': 1, 'zero_padding': 0, 'filter_number': 10
        , 'filter_distribution_interval': (-1e-2, 1e-2)}))
    layersBuilder.addLayer((LayerType.POOLING, {}))
    layersBuilder.addLayer((LayerType.FLAT, {}))
    layersBuilder.addLayer((LayerType.HIDDEN, {'activation' : NonActivation()}))
    layersBuilder.addLayer((LayerType.HIDDEN, {'activation' : NonActivation()}))


    # training params
    STEP_SIZE = 1e-4
    FILTER_STEP_SIZE = 1e-4
    REG = 1e-2
    hyperParams = HyperParams(STEP_SIZE, FILTER_STEP_SIZE, REG)

    # build layers and model
    layers = layersBuilder.build(hyperParams, inputDims, 100, 10)
    model = Model(layers, SoftMax(hyperParams), batchSize, iterations=70)


    # model getting trained
    model.train(trainingDataset, validatingDataset)
    return model



def trainWithMnist():
    data = initMNISTDataset()
    doTheStuff(data)

trainWithMnist()
