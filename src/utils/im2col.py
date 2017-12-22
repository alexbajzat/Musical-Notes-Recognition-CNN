import numpy as np

'''
    :param

    x - the input
    filterH - the height of the feature
    filterW - the width of the feature
    padding - the amount of 0 added across each axes
    stride - the step size

    :return
    each matrix reduced to blocks of pixels based on params


    I`ve reimplemented this function for some reasons:
        less validates
        working only on grayscale images
'''


def im2col(x, filterH, filterW, padding=1, stride=1):
    inputSize, height, width = x.shape
    # tuples of axis in the pixel maps to get padded
    if (height + 2 * padding - filterH) % stride == 0 or (width + 2 * padding - filterW) % stride == 0:
        raise ValueError(" The combination of filter height, width, padding and stride do not match ")

    pad = (0, 0), (padding, padding), (padding, padding)

    paddedX = np.pad(x, pad, mode='constant')

    out_height = (height + 2 * padding - filterH) / stride + 1
    out_width = (width + 2 * padding - filterW) / stride + 1
