#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Checking the implementation of MLP
Creates two concentric circles and checks whether
points near the middle circle are correctly classified.

@author: avaldes
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

#import tf_MLP_v3
#reload(tf_MLP_v3)
from tf_MLP_v3 import MLP

def sigmoid(z):
    cp_z = np.copy(z)

    mask_minus = cp_z < 0
    mask_plus = cp_z >= 0

    exp_minus = np.exp(cp_z[cp_z < 0])
    exp_plus = np.exp(-cp_z[cp_z >= 0])

    cp_z[mask_minus] = exp_minus / (1 + exp_minus)
    cp_z[mask_plus] = 1 / (1 + exp_plus)

    return cp_z


# create data

nb_black = 50
nb_red = 50
nb_data = nb_black + nb_red

s = np.linspace(0, 2*np.pi, nb_black)

x_black = np.vstack([np.cos(s), np.sin(s)]).T +\
          np.random.randn(nb_black, 2) * 0
x_red = np.vstack([2*np.cos(s), 2*np.sin(s)]).T +\
        np.random.randn(nb_red, 2) * 0

x_middle = (x_black + x_red) / 2

x_data = np.vstack((x_black, x_red))
t_data = np.asarray([0, 1]*nb_black
                    + [1, 0]*nb_red).reshape(nb_data, 2)

D = x_data.shape[1]
K = 2 

K_list = [D, 20, 20, 20, K]  # list of dimensions of layers

activation_functions = ['tanh'] * 4 + ['softmax']

mlp = MLP(K_list, activation_functions)

mlp.train(x_data, t_data,
          nb_epochs=500, batch_size=5)


dr = 0.1
r1 = 3/2 - dr
r2 = 3/2 + dr

s_test = np.linspace(0, 2*np.pi, 100)
black_pts_test = r1 * np.array([np.cos(s_test), np.sin(s_test)]).T
red_pts_test = r2 * np.array([np.cos(s_test), np.sin(s_test)]).T


y0 = mlp.predict(black_pts_test)
wrong_black = (y0[:, 0] > 1/2).squeeze()
print('Points misclassified as black:%f' % np.sum(wrong_black))

y1 = mlp.predict(red_pts_test)
wrong_red = (y1[:, 0] < 1/2).squeeze()
print('Points misclassified as red:%f' % np.sum(wrong_red))


delta = 0.01
x = np.arange(-3, 3, delta)
y = np.arange(-3, 3, delta)


X, Y = np.meshgrid(x, y)
x_pts = np.vstack((X.flatten(), Y.flatten())).T

grid_size = X.shape[0]
Z = mlp.predict(x_pts)
Z = Z.reshape(grid_size, grid_size, 2)

plt.axis('equal')
plt.contourf(X, Y, Z[:, :, 0], 50)
plt.scatter(x_black[:, 0], x_black[:, 1],
            marker=',',
            s=1,
            color='black')
plt.scatter(x_red[:, 0], x_red[:, 1],
            marker=',',
            s=1,
            color='red')

plt.scatter(x_middle[:, 0], x_middle[:, 1], marker='.', color='white')
plt.scatter(black_pts_test[:, 0], black_pts_test[:, 1],
            marker='+', color='black')
plt.scatter(black_pts_test[wrong_black, 0],
            black_pts_test[wrong_black, 1],
            marker='+', color='red')
plt.scatter(red_pts_test[:, 0], red_pts_test[:, 1],
            marker='+', color='red')
plt.scatter(red_pts_test[wrong_red, 0],
            red_pts_test[wrong_red, 1],
            marker='+', color='black')


plt.show()
