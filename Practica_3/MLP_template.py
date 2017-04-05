#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import numpy as np


class MLP(object):

    def __init__(self, K_list,
                 activation_functions, diff_activation_functions,
                 init_seed=None):

        self.K_list = K_list
        self.nb_layers = len(K_list) - 1

        self.activation_functions = activation_functions
        self.diff_activation_functions = diff_activation_functions

        self.init_seed = init_seed

        self.weights_list = None
        self.biases_list = None

        self.grad_w_list = None
        self.grad_b_list = None

        self.grad_w_averages = [0] * self.nb_layers
        self.gradsquare_w_averages = [0] * self.nb_layers
        self.grad_b_averages = [0] * self.nb_layers
        self.gradsquare_b_averages = [0] * self.nb_layers

        self.delta_w2 = [0] * self.nb_layers
        self.delta_b2 = [0] * self.nb_layers

        self.activations = None
        self.units = None
        self.y = None

        self.init_weights()

    @staticmethod
    def sigmoid(z):
        pass

    @staticmethod
    def dsigmoid(z):
        pass

    @staticmethod
    def dtanh(z):
        pass

    @staticmethod
    def relu(z):
        pass

    @staticmethod
    def drelu(z):
        pass

    @staticmethod
    def identity(z):
        pass

    @staticmethod
    def didentity(z):
        pass

    @staticmethod
    def softmax(z):
        pass

    @staticmethod
    def binary_cross_entropy(y, t_data):
        pass
    
    @staticmethod
    def softmax_cross_entropy(y, t_data):
        pass

    @staticmethod
    def cost_L2(y, t_data):
        pass

    def init_weights(self):

        if self.init_seed:
            np.random.seed(self.init_seed)

        self.weights_list = None
        self.biases_list = None

        self.grad_w_list = None
        self.grad_b_list = None

        self.grad_w_averages = [0] * self.nb_layers
        self.gradsquare_w_averages = [0] * self.nb_layers
        self.grad_b_averages = [0] * self.nb_layers
        self.gradsquare_b_averages = [0] * self.nb_layers

        weights_list = []
        biases_list = []

        for layer in range(self.nb_layers):
            new_W = np.random.randn(self.K_list[layer], self.K_list[layer + 1])
            new_b = np.zeros(self.K_list[layer + 1])
            weights_list.append(new_W)
            biases_list.append(new_b)

        self.weights_list = weights_list
        self.biases_list = biases_list

    def get_activations_and_units(self, x, parameters=None):
        pass

    def get_gradients(self, x, t, beta=0, parameters=None):
        pass

    def train(self, x_data, t_data,
              epochs=1000, batch_size=20,
              initialize_weights=False,
              eta=0.1,
              beta=0,
              gamma=0.9,
              beta_1=0.9,
              beta_2=0.999,
              epsilon=1e-8,
              method='SGD',
              print_cost=False):

        if initialize_weights:
            self.init_weights()
            pass
        
        def SGD_update():
            pass

        def momentum_update():
            pass

        def adagrad_update():
            pass

        def RMSprop_update():
            pass

        def adadelta_update():
            pass

        def nesterov_update():
            pass

        def adam_update():
            pass

        optimizers_dict = {'SGD': SGD_update,
                           'adam': adam_update,
                           'adagrad': adagrad_update,
                           'RMS_prop': RMSprop_update,
                           'adadelta': adadelta_update,
                           'momentum': momentum_update,
                           'nesterov': nesterov_update}

        
        index_list = np.arange(nb_data)
        t = 1
        
        for epoch in range(epochs):
            np.random.shuffle(index_list)
            for batch in range(nb_batches):
                batch_indices = index_list[batch*batch_size:
                                           (batch + 1)*batch_size]
                x_batch = x_data[batch_indices]
                t_batch = t_data[batch_indices]

                self.get_gradients(x_batch, t_batch, beta)

                if method == 'nesterov':
                    optimizers_dict[method]()
                else:
                    for k in range(self.nb_layers):
                        optimizers_dict[method]()
                    t = t + 1

            if print_cost:
                self.get_activations_and_units(x_data)
                if self.activation_functions[-1] == MLP.sigmoid:
                    cost = MLP.binary_cross_entropy(self.y, t_data),
                    sys.stdout.write('CREcost = % f, epoch = %d\r'
                                     % (cost, epoch))
                    sys.stdout.flush()
                elif self.activation_functions[-1] == MLP.softmax:
                    cost = MLP.softmax_cross_entropy(self.y, t_data)
                    sys.stdout.write('cost = % f, epoch = %d \r'
                                     % (cost, epoch))
                    sys.stdout.flush()
                else:
                    cost = MLP.cost_L2(self.y, t_data)
                    sys.stdout.write('cost = % f, epoch = %d\r'
                                     % (cost, epoch))
                    sys.stdout.flush()
