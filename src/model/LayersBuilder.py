from src.data.constants import LayerType
from src.model.Layers import ConvLayer, PoolLayer, FlattenLayer, np, HiddenLayer


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
                totalDepth *= config[1]['filter_number']
            if config[0] == LayerType.POOLING:
                poolingN += 1
            if config[0] == LayerType.HIDDEN:
                hiddenN += 1

        layers = []
        for config in self.__layersConfig:
            if config[0] == LayerType.CONV:
                layers.append((ConvLayer(params=config[1], hyperParams=hyperParams,
                                         activation=config[1]['activation']),
                               LayerType.CONV))
            elif config[0] == LayerType.POOLING:
                layers.append((PoolLayer(), LayerType.POOLING))

            elif config[0] == LayerType.FLAT:
                layers.append((FlattenLayer(), LayerType.FLAT))
            elif config[0] == LayerType.HIDDEN:
                if not hiddenLayerPresent:
                    inputShrink = np.power(2, poolingN)
                    fHiddenInput = int(inputDimensions[0] * inputDimensions[1] / np.power(inputShrink, 2) * totalDepth)
                    layers.append((HiddenLayer(fHiddenInput,
                                               fullyConnectedN, config[1]['activation'], hyperParams),
                                   LayerType.HIDDEN))
                    hiddenLayerPresent = True
                elif hiddenN > 1:
                    layers.append((HiddenLayer(fullyConnectedN, fullyConnectedN, config[1]['activation'], hyperParams),
                                   LayerType.HIDDEN))
                else:
                    layers.append((HiddenLayer(fullyConnectedN, outputClasesN, config[1]['activation'], hyperParams),
                                   LayerType.HIDDEN))
                hiddenN -= 1
        return layers
