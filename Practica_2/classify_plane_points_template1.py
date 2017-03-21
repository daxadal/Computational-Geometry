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

i_min = 0
j_min = 0
data_min = 1000000
middle_min = data_min

for j in range(10, 51):
    for i in range(10, 51):
        K_list = [D, i, j, K] #list of dimensions of layers


        activation_functions = [np.tanh,
                                np.tanh,
                                MLP.sigmoid]
                                
        diff_activation_functions = [MLP.dtanh,
                                     MLP.dtanh,
                                     MLP.dsigmoid]


        #%%
        x_middle = (x_black + x_red) / 2
        nb_middle = x_middle.shape[0]
        t_middle = np.asarray([0.5] * nb_middle).reshape(nb_middle, 1)

        mlp = MLP(K_list, activation_functions, diff_activation_functions)

        for k in range(20):
            mlp.train(x_data, t_data,
                       epochs=10, batch_size=10,
                       epsilon=0.1,
                       print_cost=True)

        mlp.get_activations_and_units(x_data)
        data_cost = mlp.binary_cross_entropy(mlp.y, t_data)
        if data_cost < data_min:
            data_min = data_cost
            i_min = i
            j_min = j
            print(i, " ", j, " ", data_min, "\n")
        #mlp.get_activations_and_units(x_middle)
        #print(mlp.binary_cross_entropy(mlp.y, t_middle))

