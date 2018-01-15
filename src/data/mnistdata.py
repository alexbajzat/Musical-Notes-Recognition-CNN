from mnist import MNIST as mnist
import numpy as np
from src.model.LabeledModel import LabeledModel


def initMNISTDataset():
    labeledData = []
    mdata = mnist('data/dataset')
    imgs, labels = mdata.load_training()

    for i in range(1000):
        labeledData.append(LabeledModel(np.expand_dims(np.asarray(imgs[i]).reshape(28, 28), 0), labels[i]))
    return labeledData
