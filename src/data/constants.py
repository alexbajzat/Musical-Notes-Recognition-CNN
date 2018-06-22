from enum import Enum


class LayerType(Enum):
    HIDDEN = 'HIDDEN',
    CONV = 'CONV',
    POOLING = 'POOLING',
    TEST = 'TEST',
    FLAT = 'FLAT'

