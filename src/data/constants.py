from enum import Enum


class LayerType(Enum):
    HIDDEN = 'hidden',
    CONV = 'conv',
    POOLING = 'pool',
    ACTIVATION = 'active',
    FLAT = 'flat'

