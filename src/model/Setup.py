from PIL import Image;
from src.model.LabeledModel import LabeledModel
from os import listdir
from os.path import isfile

ROOT = '../../dataset/processed/'
EXTENSION = '.JPG'
PROCESSED_RESCALE = 200, 200

''' initializing dataset from root folder
    the label is given by the directory name
    root and extension of data must be configured before initialize
'''
def initDataset():
    labeledData = []
    for folder in listdir(ROOT):
        for label in listdir(ROOT):
            url = ROOT + folder + "/" + label + EXTENSION
            if(isfile(url)):
                print('loading file: ' + url)
                img = Image.open(url).convert('LA')
                labeledData.append(LabeledModel(img, label))
    return labeledData


save = initDataset()
print(save)
