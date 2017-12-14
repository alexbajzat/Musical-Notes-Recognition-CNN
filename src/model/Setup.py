from PIL import Image;
from src.model.LabeledModel import LabeledModel
from os import listdir
from os.path import isfile

root = '../../dataset/'
extension = '.JPG'


def initDataset():
    roots = ['1', '2']
    labeledData = []
    for folder in listdir(root):
        print(folder)
        for item in listdir(root):
            url = root + folder + "/" + item + extension
            print(url)
            if(isfile(url)):
                img = Image.open(url).convert('LA')
                labeledData.append(LabeledModel(img, item))
    return labeledData


save = initDataset()
print(save)
