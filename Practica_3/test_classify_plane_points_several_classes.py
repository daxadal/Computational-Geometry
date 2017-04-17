#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: avaldes
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from MLP_v3 import MLP


nb_black = 50
nb_red = 50
nb_green = 50

nb_data = nb_black + nb_red + nb_green

s = np.linspace(0, 2*np.pi, nb_black)

x_black = np.vstack([np.cos(s), np.sin(s)]).T
x_red = 2 * np.vstack([np.cos(s), np.sin(s)]).T
x_green = 3 * np.vstack([np.cos(s), np.sin(s)]).T

x_data = np.vstack((x_black, x_red, x_green))

t_list = [1, 0, 0] * nb_black + [0, 1, 0] * nb_red + [0, 0, 1] * nb_green
t_data = np.asarray(t_list).reshape(nb_data, 3)

D = x_data.shape[1]
K = 3

K_list = [D, 100, 50, K]

activation_functions = [MLP.relu] * 2 + [MLP.softmax]
diff_activation_functions = [MLP.drelu] * 2

methods = ['SGD', 'momentum', 'nesterov', 'adagrad',
           'adadelta', 'RMS_prop', 'adam']

fig, ax = plt.subplots(2, 4)

list_pairs = [(r, c) for r in range(2) for c in range(4)]

for counter, method in enumerate(methods):
    method = methods[counter]

    mlp = MLP(K_list,
              activation_functions,
              diff_activation_functions,
              init_seed=5)

    mlp.train(x_data, t_data,
              epochs=200, batch_size=20,
              eta=0.01,
              method=method,
              print_cost=True,
              initialize_weights=True)

    delta = 0.05
    a, b = -4, 4
    x = np.arange(a, b, delta)
    y = np.arange(a, b, delta)

    X, Y = np.meshgrid(x, y)
    x_pts = np.vstack((X.flatten(), Y.flatten())).T
    mlp.get_activations_and_units(x_pts)

    grid_size = X.shape[0]
    print(method)
    r, c = list_pairs[counter]
    curr_axes = ax[r, c]
    curr_axes.axis('equal')
    curr_axes.scatter(X, Y, facecolors=mlp.y)

    curr_axes.scatter(x_data[:, 0], x_data[:, 1],
                      marker='o',
                      s=1,
                      color='black')
    
    curr_axes.set_xlim(-4, 4)
    curr_axes.set_ylim(-4, 4)
    curr_axes.set_title(method)

    mlp.get_activations_and_units(x_data)
    error = mlp.softmax_cross_entropy(mlp.y, t_data)
    curr_axes.set_xlabel('error= %.3f' % error)

plt.show()
