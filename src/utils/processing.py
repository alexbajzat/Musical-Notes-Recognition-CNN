from copy import deepcopy

import numpy as np
from PIL import Image
import datetime


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (int)((H + 2 * padding - field_height) / stride + 1)
    out_width = (int)((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    if (len(x.shape) == 3):
        x = np.expand_dims(x, axis=1)

    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def scaleBetweenValues(array, lowerBound=0, upperBound=1, dtype=int):
    min = np.min(array)
    max = np.amax(array)

    normalized = array[:]
    normalized -= min
    normalized *= ((upperBound - lowerBound) / (max - min) + lowerBound)
    return np.ndarray.astype(normalized, dtype)


'''
    save feature map of conv as pngs
'''


def exportPNGs(featured, opType):
    for img in featured:
        copy = deepcopy(img)
        copy = scaleBetweenValues(copy, 0, 255)
        fromarray = Image.fromarray(copy)
        grayscale = fromarray.convert('L')
        resized = grayscale.resize((100, 100))
        now = datetime.datetime.now()
        resized.save(
            '../features/' + str(now.strftime("%Y-%m-%d-%Hhh%Mmm")) + '-' + opType + "-" + str(id(img)) + '.png')


'''
    export model`s training as html
'''


def exportHistory(export):
    history, configuration = export
    now = datetime.datetime.now()
    file = open('../history/' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.html', 'w+')

    file.write('<h2> configuration </h2>')
    file.write('<table style="border:1px solid black;" cellpadding="10">')
    file.write('<tr><th>EPOCH</th><th>LOSS</th><th>ACCURACY</th></tr>')
    epoch = 1
    for step in history:
        file.write('<tr>')
        file.write('<td>' + str(epoch) + '</td>')
        file.write('<td>' + str(step[0]) + '</td>')
        file.write('<td>' + str(int(step[1] * 100))+ "%" + '</td>')
        file.write('</tr>')
        epoch += 1
    file.write('</table>')
