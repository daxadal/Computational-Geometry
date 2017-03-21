#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Checking the implementation of MLP
Data in points along two logarithmic spirals
@author: avaldes
"""

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import time

from MLP import MLP

# create data

nb_black = 100
nb_red = 100
nb_data = nb_black + nb_red

s = np.linspace(0, 4*np.pi, nb_black)

x_black = np.vstack([np.log(1 + s) * np.cos(s),
                     np.log(1 + s) * np.sin(s)]).T
x_red = np.vstack([-np.log(1 + s) * np.cos(s),
                   -np.log(1 + s) * np.sin(s)]).T

x_data = np.vstack((x_black, x_red))

t_data = np.asarray([0]*nb_black + [1]*nb_red).reshape(nb_data, 1)


#  Net structure

D = x_data.shape[1]  # initial dimension
K = 1  # final dimension

#  You must find the best MLP structure in order to
#  misclassify as few points as possible. Training time will be
#  measured too.
#  You can use at most 3000 weights and 2000 epochs.
#  For example:

K_list = [D, 50, 50, K]  # list of dimensions of layers
# The bigger the amount of neurons, the "smoother" it will appear,
# although there is no significant change in the amont of correctly
# classified points, which seem to be inevitably all but one of the
# centermost points (even with as few as 50 epochs, with something like
# 50 neurons in each hidden layer, or with as few as 25 neurons (or less!)
# per layer with sufficient epochs).

activation_functions = [lambda x: np.sin(x),
                        MLP.sigmoid,
                        MLP.sigmoid]

diff_activation_functions = [lambda x: np.cos(x),
                             MLP.dsigmoid,
                             MLP.dsigmoid]

# network training

time_begin = time.time()

mlp = MLP(K_list,
          activation_functions, diff_activation_functions)

mlp.train(x_data, t_data,
          epochs=2000, batch_size=10,
          epsilon=0.1,
          print_cost=True)

time_end = time.time()

print('Time used in training %f' % (time_end - time_begin))


# check if circles nearby the middle one are
# correctly classified


mlp.get_activations_and_units(x_black)
wrong_black = (mlp.y > 1/2).squeeze()
print('Points misclassified as black: {}'.format(np.sum(wrong_black)))

mlp.get_activations_and_units(x_red)
wrong_red = (mlp.y < 1/2).squeeze()
print('Points misclassified as red: {}'.format(np.sum(wrong_red)))


# plot the probability mapping and the data

delta = 0.01
x = np.arange(-3, 3, delta)
y = np.arange(-3, 3, delta)


X, Y = np.meshgrid(x, y)
x_pts = np.vstack((X.flatten(), Y.flatten())).T
mlp.get_activations_and_units(x_pts)
grid_size = X.shape[0]
Z = mlp.y.reshape(grid_size, grid_size)

plt.axis('equal')
plt.contourf(X, Y, Z, 50)
plt.scatter(x_black[:, 0], x_black[:, 1],
            marker=',',
            s=1,
            color='black')
plt.scatter(x_red[:, 0], x_red[:, 1],
            marker=',',
            s=1,
            color='red')


plt.scatter(x_black[wrong_black, 0],
            x_black[wrong_black, 1],
            facecolors='None',
            s=50,
            marker='o', color='red')

plt.scatter(x_red[wrong_red, 0],
            x_red[wrong_red, 1],
            facecolors='None',
            s=50,
            marker='o', color='black')


plt.show()
