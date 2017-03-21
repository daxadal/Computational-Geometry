#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 23:13:44 2017

@author: avaldes
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from MLP import MLP

def f(x):
    return np.sin(x)

N = 40
a, b = 0, 2*np.pi
x_data = np.linspace(a, b, N).reshape(N, 1)
t_data = f(x_data)

D = 1
K = 1

K_list = [D, 20, 20, K] #list of dimensions



activation_functions = [MLP.relu,
                        MLP.sigmoid,
                        MLP.identity]
diff_activation_functions = [MLP.drelu,
                             MLP.dsigmoid,
                             MLP.didentity]

"""
The results of the previous configuration vary a lot.

A single hidden layer (as considered by the universal approximation theorem)
yields also very different results, and increasing the amount of neurons does
no greatly improve the result. The configuration would be:

K_list = [D, 20, K] #list of dimensions



activation_functions = [MLP.sigmoid,
                        MLP.identity]
diff_activation_functions = [MLP.dsigmoid,
                             MLP.didentity]

As expected, the following configuration yields an almost perfect result:

K_list = [D, 1, K] #list of dimensions



activation_functions = [lambda x: np.sin(x),
                        MLP.identity]
diff_activation_functions = [lambda x: np.cos(x),
                             MLP.didentity]
"""

mlp = MLP(K_list, activation_functions, diff_activation_functions)

mlp.train(x_data, t_data, epochs=1000, batch_size=1, epsilon=0.1, beta=0.001)
    
M = 200
x_pts = np.linspace(a - 5, b + 5, M).reshape(M, 1)
mlp.get_activations_and_units(x_pts)

plt.plot(x_pts, mlp.y, color='black')
plt.plot(x_data, t_data)
plt.show()
