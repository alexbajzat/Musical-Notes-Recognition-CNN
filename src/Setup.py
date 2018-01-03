from PIL import Image
from src.model.LabeledModel import LabeledModel
from os import listdir
from os.path import isfile
import numpy as np

DATASET_ROOT = '../dataset/processed/'
REAL_SET_ROOT = '../images'
PROCESSED_RESIZE = 64 , 64

''' initializing dataset from root folder
    the label is given by the directory name
    root and extension of data must be configured before initialize
'''
def initDataset():
    labeledData = []
    for folder in listdir(DATASET_ROOT):
        for filename in listdir(DATASET_ROOT + folder):
            url = DATASET_ROOT + folder + "/" + filename
            print(url)
            if(isfile(url)):
                print('loading file: ' + url)
                img = Image.open(url).convert('L')
                img= img.resize(PROCESSED_RESIZE, Image.ANTIALIAS)
                img = np.asarray(img.getdata()).reshape(img.size[0], img.size[1])
                labeledData.append(LabeledModel(img, folder))
    return labeledData

def loadImages():
        images = []
        for filename in listdir(DATASET_ROOT):
            url = DATASET_ROOT + "/" + filename
            print(url)
            if (isfile(url)):
                print('loading file: ' + url)
                img = Image.open(url).convert('L')
                img = img.resize(PROCESSED_RESIZE, Image.ANTIALIAS)
                parsed = np.asarray(img.getdata()).reshape(img.size[0], img.size[1])
                images.append(parsed, img)
        return images







