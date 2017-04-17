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
        # ->
        return 1 / (1 + np.exp(-z))
        # <-

    @staticmethod
    def dsigmoid(z):
        # ->
        return MLP.sigmoid(z)*(1 - MLP.sigmoid(z))
        # <-

    @staticmethod
    def dtanh(z):
        # ->
        return 1 - np.tanh(z)**2
        # <-

    @staticmethod
    def relu(z):
        # ->
        ret = np.copy(z)
        ret[z<0] = 0
        return ret
        # <-

    @staticmethod
    def drelu(z):
        # ->
        ret = np.copy(z)
        ret[z<0] = 0
        ret[z>=0] = 1
        return ret
        # <-

    @staticmethod
    def identity(z):
        # ->
        return z
        # <-

    @staticmethod
    def didentity(z):
        # ->
        return np.ones(z.shape())
        # <-

    @staticmethod
    def softmax(z):
        # ->
        exps = np.exp(z)
        exps_sums = np.sum(exps, axis=1)
        return exps / exps_sums[:, np.newaxis]
        # <-

    @staticmethod
    def binary_cross_entropy(y, t_data):
        # ->
        return -np.sum(t_data * np.log(y) + (1 - t_data) * np.log(1 - y))
        # <-
    
    @staticmethod
    def softmax_cross_entropy(y, t_data):
        # ->
        return -np.sum(t_data * np.log(y))
        # <-

    @staticmethod
    def cost_L2(y, t_data):
        # ->
        return np.sum((y-t_data)**2) / 2
        # <-

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
        # ->
        if parameters == None:
            weights_list = self.weights_list
            biases_list = self.biases_list
        else:
            weights_list, biases_list = parameters
        
        activations = [x]
        units = [x]

        for i in range(self.nb_layers):
            W = weights_list[i]
            b = biases_list[i]
            h = self.activation_functions[i]
            activations.append(units[i].dot(W) + b)
            units.append(h(activations[i + 1]))
        
        self.activations = activations
        self.units = units
        self.y = units[-1]
        # <-

    def get_gradients(self, x, t, beta=0, parameters=None):
        # ->
        if parameters == None:
            weights_list = self.weights_list
            biases_list = self.biases_list
        else:
            weights_list, biases_list = parameters
        
        self.get_activations_and_units(x, parameters=parameters)
        
        N = x.shape[0]
        
        delta = self.y-t
        grad_w = np.einsum('ni,nj->nji', delta, self.units[-2])
        grad_w_list = [np.sum(grad_w, axis=0)/N]
        grad_b_list = [np.sum(delta, axis=0)/N]
        for k in range(self.nb_layers-2, -1, -1):
            delta = np.einsum('ni,ij,nj->ni',
                              self.diff_activation_functions[k](self.activations[k+1]),
                              weights_list[k+1],
                              delta)
            grad_w = np.einsum('ni,nj->nji', delta, self.units[k])
            grad_w_list.insert(0, np.sum(grad_w, axis=0)/N)
            grad_b_list.insert(0, np.sum(delta, axis=0)/N)
        if beta != 0:
            for i in range(self.nb_layers):
                grad_w_list[i] += beta * weights_list[i]
                grad_b_list[i] += beta * biases_list[i]

        self.grad_w_list = grad_w_list
        self.grad_b_list = grad_b_list
        # <-

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
            # ->
            self.weights_list[k] -= eta * self.grad_w_list[k]
            self.biases_list[k] -= eta * self.grad_b_list[k]
            # <-

        def momentum_update():
            # ->
            self.grad_w_averages[k] *= gamma
            self.grad_w_averages[k] += eta * self.grad_w_list[k]
            self.grad_b_averages[k] *= gamma
            self.grad_b_averages[k] += eta * self.grad_b_list[k]

            self.weights_list[k] -= self.grad_w_averages[k]
            self.biases_list[k] -= self.grad_b_averages[k]
            # <-

        def adagrad_update():
            # ->
            self.gradsquare_w_averages[k] += self.grad_w_list[k]**2
            self.gradsquare_b_averages[k] += self.grad_b_list[k]**2
            
            new_eta = eta / np.sqrt(self.gradsquare_w_averages[k] + epsilon)
            self.weights_list[k] -= new_eta * self.grad_w_list[k]
            new_eta = eta / np.sqrt(self.gradsquare_b_averages[k] + epsilon)
            self.biases_list[k] -= new_eta * self.grad_b_list[k]
            # <-

        def RMSprop_update():
            # ->
            self.gradsquare_w_averages[k] *= gamma
            self.gradsquare_w_averages[k] += (1 - gamma) * self.grad_w_list[k]**2
            self.gradsquare_b_averages[k] *= gamma
            self.gradsquare_b_averages[k] += (1 - gamma) * self.grad_b_list[k]**2
            
            new_eta = eta / np.sqrt(self.gradsquare_w_averages[k] + epsilon)
            self.weights_list[k] -= new_eta * self.grad_w_list[k]
            new_eta = eta / np.sqrt(self.gradsquare_b_averages[k] + epsilon)
            self.biases_list[k] -= new_eta * self.grad_b_list[k]
            # <-

        def adadelta_update():
            # ->
            self.gradsquare_w_averages[k] *= gamma
            self.gradsquare_w_averages[k] += (1 - gamma) * self.grad_w_list[k]**2
            self.gradsquare_b_averages[k] *= gamma
            self.gradsquare_b_averages[k] += (1 - gamma) * self.grad_b_list[k]**2
            
            delta_w = np.sqrt(self.delta_w2[k] + epsilon) 
            delta_w /= np.sqrt(self.gradsquare_w_averages[k] + epsilon)
            delta_w *= self.grad_w_list[k]
            self.weights_list[k] -= delta_w
            
            delta_b = np.sqrt(self.delta_b2[k] + epsilon)
            delta_b /= np.sqrt(self.gradsquare_b_averages[k] + epsilon)
            delta_b *= self.grad_b_list[k]
            self.biases_list[k] -= delta_b
            
            self.delta_w2[k] *= gamma
            self.delta_w2[k] += (1 - gamma) * delta_w**2
            self.delta_b2[k] *= gamma
            self.delta_b2[k] += (1 - gamma) * delta_b**2
            # <-

        def nesterov_update():
            # ->
            new_weights = []
            new_biases = []
            
            for k in range(self.nb_layers):
                self.grad_w_averages[k] *= gamma
                self.grad_b_averages[k] *= gamma
                
                new_weights.append(self.weights_list[k] - self.grad_w_averages[k])
                new_biases.append(self.biases_list[k] - self.grad_b_averages[k])

            params = (new_weights, new_biases)
            self.get_gradients(x_batch, t_batch, beta, parameters=params)

            for k in range(self.nb_layers):
                self.grad_w_averages[k] += eta * self.grad_w_list[k]
                self.grad_b_averages[k] += eta * self.grad_b_list[k]

                self.weights_list[k] -= self.grad_w_averages[k]
                self.biases_list[k] -= self.grad_b_averages[k]
            # <-

        def adam_update():
            # ->
            self.grad_w_averages[k] *= beta_1
            self.grad_w_averages[k] += (1 - beta_1) * self.grad_w_list[k]
            self.grad_b_averages[k] *= beta_1
            self.grad_b_averages[k] += (1 - beta_1) * self.grad_b_list[k]

            self.gradsquare_w_averages[k] *= beta_2
            self.gradsquare_w_averages[k] += (1 - beta_2) * self.grad_w_list[k]**2
            self.gradsquare_b_averages[k] *= beta_2
            self.gradsquare_b_averages[k] += (1 - beta_2) * self.grad_b_list[k]**2

            m_w = self.grad_w_averages[k] / (1 - beta_1**t)
            m_b = self.grad_b_averages[k] / (1 - beta_1**t)
            v_w = self.gradsquare_w_averages[k] / (1 - beta_2**t)
            v_b = self.gradsquare_b_averages[k] / (1 - beta_2**t)

            self.weights_list[k] -= (eta / (np.sqrt(v_w) + epsilon)) * m_w
            self.biases_list[k] -= (eta / (np.sqrt(v_b) + epsilon)) * m_b
            # <-

        optimizers_dict = {'SGD': SGD_update,
                           'adam': adam_update,
                           'adagrad': adagrad_update,
                           'RMS_prop': RMSprop_update,
                           'adadelta': adadelta_update,
                           'momentum': momentum_update,
                           'nesterov': nesterov_update}

        # ->
        nb_data = x_data.shape[0]
        # <-
        index_list = np.arange(nb_data)
        t = 1
        
        # ->
        nb_batches = nb_data // batch_size
        # <-
        for epoch in range(epochs):
            np.random.shuffle(index_list)
            for batch in range(nb_batches):
                batch_indices = index_list[batch*batch_size:
                                           (batch + 1)*batch_size]
                x_batch = x_data[batch_indices]
                t_batch = t_data[batch_indices]

                # self.get_gradients(x_batch, t_batch, beta)

                if method == 'nesterov':
                    optimizers_dict[method]()
                else:
                    # ->
                    self.get_gradients(x_batch, t_batch, beta)
                    # <-
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
