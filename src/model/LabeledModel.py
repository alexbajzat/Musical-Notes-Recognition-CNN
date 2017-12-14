class LabeledModel(object):
    def __init__(self, data, label):
        self.__data = data
        self.__label = label

    def getData(self):
        return self.__data

    def getLabel(self):
        return self.__label
