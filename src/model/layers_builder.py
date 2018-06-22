from enum import Enum

from src.data.constants import LayerType
from src.model.activations import ReLUActivation, NonActivation
from src.model.layers import ConvLayer, PoolLayer, FlattenLayer, np, HiddenLayer, TestingLayer

from src.utils.processing import parseJSON

ACTIVATIONS_MAP = {'RELU': ReLUActivation(), "NON": NonActivation()}


class LayersBuilder(object):
    def __init__(self):
        self.__layersConfig = []

    def addLayer(self, config):
        self.__layersConfig.append(config)

    def build(self, hyperParams, inputDimensions, fullyConnectedN, outputClasesN):
        totalDepth = 1
        poolingN = 0
        hiddenN = 0
        hiddenLayerPresent = False
        for config in self.__layersConfig:
            if config[0] == LayerType.CONV:
                totalDepth *= config[1].filter_number
            if config[0] == LayerType.POOLING:
                poolingN += 1
            if config[0] == LayerType.HIDDEN:
                hiddenN += 1
        inputShrink = np.power(2, poolingN)
        fHiddenInput = int(inputDimensions[0] * inputDimensions[1] / np.power(inputShrink, 2) * totalDepth)
        layers = []
        for config in self.__layersConfig:
            if config[0] == LayerType.CONV:
                layers.append((ConvLayer(params=config[1], hyperParams=hyperParams,
                                         activation= ACTIVATIONS_MAP[config[1].activation]),
                               LayerType.CONV))
            elif config[0] == LayerType.POOLING:
                layers.append((PoolLayer(), LayerType.POOLING))
            elif config[0] == LayerType.FLAT:
                layers.append((FlattenLayer(), LayerType.FLAT))
            elif config[0] == LayerType.HIDDEN:
                if not hiddenLayerPresent:
                    layers.append((HiddenLayer(fHiddenInput,
                                               fullyConnectedN, ACTIVATIONS_MAP[config[1].activation], hyperParams),
                                   LayerType.HIDDEN))
                    hiddenLayerPresent = True
                elif hiddenN > 1:
                    layers.append((HiddenLayer(fullyConnectedN, fullyConnectedN, ACTIVATIONS_MAP[config[1].activation], hyperParams),
                                   LayerType.HIDDEN))
                else:
                    layers.append((HiddenLayer(fullyConnectedN, outputClasesN, ACTIVATIONS_MAP[config[1].activation], hyperParams),
                                   LayerType.HIDDEN))
                hiddenN -= 1
            elif config[0] == LayerType.TEST:
                layers.append((TestingLayer(fHiddenInput,outputClasesN), LayerType.TEST))
        return layers

    def reconstruct(self, modelJson):
        layers = []
        data = parseJSON(modelJson)
        for layer in data.model.layers:
            print(layer, '\n')
            if(layer.type == 'CONV'):
                if(layer.activation == 'RELU'):
                    activation = ReLUActivation()
                else:
                    activation = NonActivation()
                layers.append((ConvLayer(activation=activation, filters=layer.weights, stride=layer.convParams.stride), LayerType.CONV))
            elif(layer.type == 'POOLING'):
                layers.append((PoolLayer(), LayerType.POOLING))
            elif(layer.type == 'FLAT'):
                layers.append((FlattenLayer(), LayerType.FLAT))
            elif(layer.type == 'HIDDEN'):
                layers.append((HiddenLayer(weights=layer.weights, biases=layer.biases), LayerType.FLAT))
        sampleData= np.asarray(data.model.sample.data)
        sampleRaw = np.asarray(data.model.sample.result)
        sampleProbabilities = np.asarray(data.model.sample.probabilities)
        return layers, (sampleData, sampleRaw, sampleProbabilities)