import mnist
import numpy as np
from src.model.labeled_model import LabeledModel


def initMNISTDataset():
    labeledData = []
    # mdata = mnist('data/dataset')
    imgs, labels = mnist.train_images(), mnist.train_labels()

    for i in range(5000):
        labeledData.append(LabeledModel(np.expand_dims(np.asarray(imgs[i]).reshape(28, 28), 0), labels[i]))
    return labeledData
