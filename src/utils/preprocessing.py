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


def im2col_indices(x, filterH, filterW, padding=1, stride=1):
    inputSize, height, width = x

    # we have no color channels
    channelSize = 1

    if (height + 2 * padding - filterH) % stride != 0 or (width + 2 * padding - filterW) % stride != 0:
        raise ValueError(" The combination of filter height, width, padding and stride do not match ")

    out_height = int((height + 2 * padding - filterH) / stride + 1)
    out_width = int((width + 2 * padding - filterW) / stride + 1)

    i0 = np.repeat(np.arange(filterH), filterW)
    i0 = np.tile(i0, channelSize)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filterW), filterH * channelSize)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channelSize), filterH * filterW).reshape(-1, 1)

    return (k, i, j)


def im2col(x, filterH, filterW, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = im2col_indices(x.shape, filterH, filterW, padding,
                             stride)

    cols = x_padded[:, i, j]
    C = x.shape[0]
    cols = cols.transpose(1, 2, 0).reshape(filterH * filterW, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, H, W = x_shape

    # hardcode the channel to 1
    C = 1

    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = im2col_indices(x_shape, field_height, field_width, padding,
                             stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
