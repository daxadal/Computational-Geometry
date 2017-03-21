#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 13:13:13 2017

@author: avaldes
"""
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from MLP import MLP

#%% Create data    
#np.random.seed(5)

nb_black = 50
nb_red = 50
nb_data = nb_black + nb_red

s = np.linspace(0, 2*np.pi, nb_black)

x_black = np.vstack([np.cos(s), np.sin(s)]).T +\
          np.random.randn(nb_black, 2) * 0
x_red = np.vstack([2*np.cos(s), 2*np.sin(s)]).T +\
        np.random.randn(nb_red, 2) * 0
x_data = np.vstack((x_black, x_red)) 

t_data = np.asarray([0]*nb_black + [1]*nb_red).reshape(nb_data, 1)    

#%% Net structure

D = x_data.shape[1] #initial dimension
K = 1 #final dimension

"""
K_list = [D, 50, 10, K] #list of dimensions of layers

activation_functions = [np.tanh,
                        np.tanh,
                        MLP.sigmoid]
                        
diff_activation_functions = [MLP.dtanh,
                             MLP.dtanh,
                             MLP.dsigmoid]

"""
# A naive automatic test for different values for the dimensions
# did not yield any significant result.

#A simpler alternative:

K_list = [D, 20, K] #list of dimensions of layers
# It works even with only two neurons in the hidden layer.

activation_functions = [lambda x: x**2,
                        MLP.sigmoid]
                        
diff_activation_functions = [lambda x: 2*x,
                             MLP.dsigmoid]

#%%
delta = 0.01
x = np.arange(-3, 3, delta)
y = np.arange(-3, 3, delta)

X, Y = np.meshgrid(x, y)

x_pts = np.vstack((X.flatten(), Y.flatten())).T
x_middle = (x_black + x_red) / 2

mlp = MLP(K_list, activation_functions, diff_activation_functions)

for k in range(1000):
    mlp.train(x_data, t_data,
               epochs=10, batch_size=10,
               epsilon=0.1,
               print_cost=True)
    
    mlp.get_activations_and_units(x_pts)
    grid_size = X.shape[0]
    Z = mlp.y.reshape(grid_size, grid_size)
    
    
    plt.axis('equal')
    plot_contour = plt.contourf(X, Y, Z, 10)
    plt.scatter(x_black[:, 0], x_black[:, 1], marker='o', color='black')
    plt.scatter(x_red[:, 0], x_red[:, 1], marker='x', color='red')
    
    plt.scatter(x_middle[:, 0], x_middle[:, 1], marker='.', color='white')

    plt.pause(0.01)
    plt.draw()
    
plt.show()
