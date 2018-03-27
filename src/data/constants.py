from enum import Enum


class LayerType(Enum):
    HIDDEN = 'hidden',
    CONV = 'conv',
    POOLING = 'pool',
    TEST = 'test',
    ACTIVATION = 'active',
    FLAT = 'flat'

