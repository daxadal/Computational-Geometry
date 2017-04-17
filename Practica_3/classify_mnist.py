from __future__ import division, print_function

import cPickle
import gzip
import sys
import numpy as np
from MLP_v3 import MLP

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

x_data = train_set[0]
x_data = x_data
mean_image = np.mean(x_data, axis=0)
x_data -= mean_image
x_data = x_data / 255

t_data = train_set[1]
nb_data = t_data.shape[0]
one_hot_tdata = np.zeros((nb_data, 10))
one_hot_tdata[np.arange(nb_data), t_data] = 1


K_list = [784, 100, 50, 10]
activation_functions = [MLP.relu] * 2 + [MLP.softmax]

diff_activation_functions = [MLP.drelu] * 2


mlp = MLP(K_list,
          activation_functions,
          diff_activation_functions,
          init_seed=5)

if 1:
    x_test, t_test = test_set
    nb_epochs = 31

    for epoch in range(nb_epochs):
        initialize_weights = (epoch == 0)
        mlp.train(x_data, one_hot_tdata,
                  epochs=1,
                  batch_size=50,
                  initialize_weights=initialize_weights,
                  eta=0.01,
                  beta=0,
                  method='adam',
                  print_cost=False)

        mlp.get_activations_and_units((x_test - mean_image) / 255)
        nb_correct = np.sum(np.equal(t_test, np.argmax(mlp.y, axis=1)))
        sys.stdout.write('fallos en los datos de test = %d, epoch= %d\r'
                         % (10000 - nb_correct, epoch))
        sys.stdout.flush()

if 0:
    np.load('./mnist_weights.npy')
    np.load('./mnist_biases.npy')

np.save('mnist_weights_v2', mlp.weights_list, allow_pickle=True)
np.save('mnist_biases_v2', mlp.biases_list, allow_pickle=True)
