from PIL import Image
from src.model.LabeledModel import LabeledModel
from os import listdir
from os.path import isfile

ROOT = '../dataset/processed/'
PROCESSED_RESIZE = 100 , 100

''' initializing dataset from root folder
    the label is given by the directory name
    root and extension of data must be configured before initialize
'''
def initDataset():
    labeledData = []
    for folder in listdir(ROOT):
        print('Parsing URL: ' +  ROOT +  folder)
        for filename in listdir(ROOT + folder):
            url = ROOT + folder + "/" + filename
            print(url)
            if(isfile(url)):
                print('loading file: ' + url)
                img = Image.open(url).convert('LA')
                img = img.resize(PROCESSED_RESIZE, Image.ANTIALIAS)
                labeledData.append(LabeledModel(img, folder))
    return labeledData


