# -*- coding: utf-8 -*-
"""
Test MLP class for regression

@author: avaldes
"""
from __future__ import division, print_function

import numpy as np
#import matplotlib.pyplot as plt

from MLP_v3 import MLP

def f1(x):
    return 1 / (1 + x**2)

def f2(x):
    return np.sin(x)


nb_data = 500
x_data = np.linspace(-5, 5, nb_data).reshape(nb_data, 1)
t_data1 = f1(x_data)
t_data2 = f2(x_data)

D = 1
K = 1

K_list = [D, 100, K]  # list of dimensions of layers

activation_functions = [MLP.sigmoid] * 1 + [MLP.identity]
diff_activation_functions = [MLP.dsigmoid] * 1

methods = ['SGD',
           'momentum',
#           'nesterov',
           'adagrad',
           'adadelta',
           'RMS_prop',
           'adam']

# methods = ['nesterov']

#fig, ax = plt.subplots(2, 7)

for t_data_nb, t_data in enumerate([t_data1, t_data2]):
    for method_nb, method in enumerate(methods):
        mlp = MLP(K_list,
                  activation_functions,
                  diff_activation_functions,
                  init_seed=6)

        print(method)
        mlp.train(x_data, t_data,
                  epochs=1000, batch_size=100,
                  eta=0.01,
                  method=method,
                  gamma=0.9,
                  beta=0,
                  beta_1=0.99,
                  beta_2=0.999,
                  initialize_weights=True,
                  print_cost=True)

        mlp.get_activations_and_units(x_data)

        error = mlp.cost_L2(mlp.y, t_data)
        """
        curr_ax = ax[t_data_nb, method_nb]

        curr_ax.plot(x_data, mlp.y)
        curr_ax.plot(x_data, t_data, ',')

        curr_ax.set_xlim(-5, 5)
        curr_ax.set_ylim(-1 * t_data_nb, 1)

        if t_data_nb == 0:
            curr_ax.set_title(method)

        curr_ax.set_xlabel('error= %.3f' % error)
        """
        print('\n')
#plt.show()
