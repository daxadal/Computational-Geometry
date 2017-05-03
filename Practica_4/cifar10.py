import os
import cPickle


import numpy as np
import matplotlib.pyplot as plt

import download
reload(download)

import net_constructor
reload(net_constructor)
from net_constructor import NetConstructor


def maybe_download_and_extract():
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    download.maybe_download_and_extract(url, '.')

def process_data(file_name):

    with open(file_name, 'rb') as file:
        dict_data = cPickle.load(file)

    x_data = dict_data['data']
    t_data = dict_data['labels']

    N = x_data.shape[0]

    x_data = x_data.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)
    t_onehot = np.zeros((N, 10))
    t_onehot[np.arange(N), t_data] = 1

    return x_data, t_onehot


def get_data():

    FILES_TRAIN = ['./cifar-10-batches-py/data_batch_{}'.format(k)
                   for k in range(1, 6)]
    FILE_TEST = './cifar-10-batches-py/test_batch'

    x_test, t_test = process_data(FILE_TEST)

    x_list = []
    t_list = []
    for file_name in FILES_TRAIN:
        x_data, t_data = process_data(file_name)
        x_list.append(x_data)
        t_list.append(t_data)

    x_data = np.vstack(x_list)
    t_data = np.vstack(t_list)

    return x_data, x_test, t_data, t_test


if __name__ == '__main__':

    maybe_download_and_extract()
    x_data, x_test, t_data, t_test = get_data()

    x_data_mean = np.mean(x_data, axis=0)

    x_data = (x_data - x_data_mean) / 255

    layer_dict = dict()

    layer_dict[0] = {'dim': (32, 32, 3)}

    layer_dict[1] = {'type': 'conv',
                     'channels': 64,
                     'kernel_size': (5, 5),
                     'stride': (1, 1),
                     'padding': 'SAME',
                     'activation': 'relu'}

    layer_dict[2] = {'type': 'maxpool',
                     'ksize': (3, 3),
                     'strides': (2, 2),
                     'padding': 'SAME'}

    layer_dict[3] = {'type': 'LRN',
                     'k': 1.0,
                     'alpha': 0.001 / 9.0,
                     'beta': 0.75}

    layer_dict[4] = {'type': 'conv',
                     'channels': 64,
                     'kernel_size': (5, 5),
                     'stride': (1, 1),
                     'padding': 'SAME',
                     'activation': 'relu'}

    layer_dict[5] = {'type': 'LRN',
                     'k': 4.0,
                     'alpha': 0.001 / 9.0,
                     'beta': 0.75}


    layer_dict[6] = {'type': 'maxpool',
                     'ksize': (3, 3),
                     'strides': (2, 2),
                     'padding': 'SAME'}

 
    layer_dict[7] = {'type': 'fc',
                     'dim': 384,
                     'activation': 'relu'}

    layer_dict[8] = {'type': 'fc',
                     'dim': 192,
                     'activation': 'relu'}

    layer_dict[9] = {'type': 'fc',
                     'dim': 10,
                     'activation': 'softmax'}

    nb_layers = len(layer_dict.keys())
    layer_list = [layer_dict[k] for k in range(nb_layers)]

    nnet = NetConstructor(layer_list)

    TRAIN = False
    if TRAIN:
        nnet.train(x_data, t_data,
                   method=('adam', {'eta': 0.001}),  # , 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-8
                   nb_epochs=1000,
                   batch_size=128)

    pred = nnet.predict((x_test - x_data_mean) / 255)

    true_classes = np.argmax(t_test, axis=1)
    pred_classes = np.argmax(pred, axis=1)


    nb_good_pred = np.sum(np.equal(true_classes, classes))

    print(nb_good_pred / 10000) # aprox 0.66 %
