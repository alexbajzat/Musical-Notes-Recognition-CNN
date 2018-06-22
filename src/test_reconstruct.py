from src.data.setup import Constants
from src.model.layers_builder import LayersBuilder
from src.neural_model import Model

name = input('Name of model file to load: ')
url = Constants.MODEL_ROOT + '/' + name
print('opening file: ' + url)

file = open(url, 'r+')

builder = LayersBuilder()
layers, sample = builder.reconstruct(file.read())
model = Model(layers)
result = model.predict(sample[0])
print()