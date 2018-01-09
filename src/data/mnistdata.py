from mnist import MNIST
import numpy as np
from src.model.LabeledModel import LabeledModel


def initMNISTDataset():
    labeledData = []
    mdata = MNIST('data/dataset')
    imgs, labels = mdata.load_training()

    for i in range(500):
        labeledData.append(LabeledModel(np.expand_dims(np.asarray(imgs[i]).reshape(28, 28), 0), labels[i]))
    return labeledData
