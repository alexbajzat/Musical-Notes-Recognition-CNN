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

    # we have no color channels
    channelSize = 0

    if (height + 2 * padding - filterH) % stride == 0 or (width + 2 * padding - filterW) % stride == 0:
        raise ValueError(" The combination of filter height, width, padding and stride do not match ")

    # tuples of axis in the pixel maps to get padded
    pad = (0, 0), (padding, padding), (padding, padding)

    paddedX = np.pad(x, pad, mode='constant')

    out_height = int((height + 2 * padding - filterH) / stride + 1)
    out_width = int((width + 2 * padding - filterW) / stride + 1)

    i0 = np.repeat(np.arange(filterH), filterW)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filterW), filterH * channelSize)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channelSize), filterH * filterW).reshape(-1, 1)

    return (k, i, j)
