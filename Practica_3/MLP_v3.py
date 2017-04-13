#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import numpy as np


class MLP(object):

    def __init__(self, K_list,
                 activation_functions, diff_activation_functions,
                 init_seed=None):

        """
        Initialize the parameters for the multilayer perceptron
        """
        self.K_list = K_list
        self.nb_layers = len(K_list) - 1

        self.activation_functions = activation_functions
        self.diff_activation_functions = diff_activation_functions

        self.init_seed = init_seed

        self.weights_list = None
        self.biases_list = None

        self.grad_w_list = None
        self.grad_b_list = None

        self.grad_w_averages = [0] * self.nb_layers  #v (en momentum), m (en adam)
        self.gradsquare_w_averages = [0] * self.nb_layers  #v (en adam)
        self.grad_b_averages = [0] * self.nb_layers  #v (en momentum), m (en adam)
        self.gradsquare_b_averages = [0] * self.nb_layers  #v (en adam)

        self.delta_w2 = [0] * self.nb_layers
        self.delta_b2 = [0] * self.nb_layers

        self.activations = None
        self.units = None
        self.y = None

        self.init_weights()

    #%% definition of activation functions and derivatives
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
        
    @staticmethod
    def dsigmoid(z):
        return MLP.sigmoid(z)*(1 - MLP.sigmoid(z))

    @staticmethod    
    def dtanh(z):
        return 1 - np.tanh(z)**2

    @staticmethod    
    def relu(z):
        #your code here
        ret = np.copy(z)
        ret[z<0] = 0
        return ret
        #--------------
        
    @staticmethod    
    def drelu(z):
        #your code here
        ret = np.copy(z)
        ret[z<0] = 0
        ret[z>=0] = 1
        return ret
        #--------------
        
    @staticmethod
    def identity(z):
        #your code here
        return z
        #--------------    
    @staticmethod
    def didentity(z):
        #your code here
        return [1]*z.shape[0]
        #--------------    
    @staticmethod
    def softmax(z):
        #your code here
        exps = np.exp(z)
        exps_sums = np.sum(exps, axis=1)
        return exps / exps_sums[:, np.newaxis]
        #--------------   
     #%% cost functions
    @staticmethod
    def binary_cross_entropy(y, t_data):
        return -np.sum(t_data * np.log(y) + (1 - t_data) * np.log(1 - y))
    
    
    @staticmethod
    def softmax_cross_entropy(y, t_data):
        #your code here
        return -np.sum(t_data * np.log(y))
        #--------------
    
    @staticmethod
    def cost_L2(y, t_data):
        #your code here
        return np.sum((y-t_data)**2)/2
        #return np.sum( np.linalg.norm(y-t_data)**2 ) / 2
        #--------------   

    def init_weights(self):
        """ 
        initialize node weights to random values and node biases to zeros
        """
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

    #%% feed forward pass
    def get_activations_and_units(self, x):
        """
        This function computes the list of activations, the list of units and the output value
        activations[i] is given by: units[i].dot(W) + b
        units[i] is given by: activation_functions[i] (activations[i + 1])
        the output value is the last elment of the list of units

        Parameters 
        ------------
        x : numpy.ndarray
            input values
        
        Examples
        --------
        >>> x = np.array([[1,2,3],[4,5,6]])
        >>> K_list = [3, 1] #final dimension
        >>> activation_functions = [MLP.sigmoid]
        >>> diff_activation_functions = [MLP.dsigmoid]
        >>> mlp = MLP(K_list, activation_functions, diff_activation_functions)
        >>> mlp.get_activations_and_units(x)
        >>> mlp.activations
        [array([[1, 2, 3],
        [4, 5, 6]]), array([[-0.32823445],
        [-0.40865953]])]
        >>> mlp.units
        [array([[1, 2, 3],
        [4, 5, 6]]), array([[ 0.52281352],
        [ 0.77669704]])]
        """
        activations = [x]
        units = [x]
        
        for i in range(self.nb_layers):
            #your code here
            W = self.weights_list[i]
            b = self.biases_list[i]
            h = self.activation_functions[i]
            activations.append(units[i].dot(W) + b)
            units.append(h(activations[i + 1]))
        y = units[-1]
            #----
        self.activations = activations 
        self.units = units
        self.y = y
    
    #%% backpropagation 
    def get_gradients(self, x, t, beta=0):
        """
        This function computes gradients of the weights and the errors using backpropagation
        
        Parameters 
        ------------
        x : numpy.ndarray
            input values
        t : numpy.ndarray
            List of correct results. "1" means red_point and "0" means black_point
        
        Examples
        --------
        >>> x = np.array([[1,2,3],[4,5,6]])
        >>> K_list = [3, 1] #final dimension
        >>> activation_functions = [MLP.sigmoid]
        >>> diff_activation_functions = [MLP.dsigmoid]
        >>> mlp = MLP(K_list, activation_functions, diff_activation_functions)
        >>> t = np.array([1,0])
        >>> mlp.get_gradients(x,t,0)
        >>> mlp.grad_w_list
        [array([[-1.69174691,  0.80825309],
        [-2.37875131,  1.12124869],
        [-3.06575572,  1.43424428]])]
        >>> mlp.grad_b_list
        [array([-0.6870044,  0.3129956])]
        """
        self.get_activations_and_units(x)
    
        N = x.shape[0]
        
        grad_w_list = []
        grad_b_list = []
    
        #your code here
        delta = self.y-t
        grad_w = np.einsum('ni,nj->nji', delta, self.units[-2])
        grad_w_list = [np.sum(grad_w, axis=0)/N]
        grad_b_list = [np.sum(delta, axis=0)/N]
        for k in range(self.nb_layers-2, -1, -1):
            delta = np.einsum('ni,ij,nj->ni',
                              self.diff_activation_functions[k](self.activations[k+1]),
                              self.weights_list[k+1],
                              delta)
            grad_w = np.einsum('ni,nj->nji', delta, self.units[k])
            grad_w_list.insert(0, np.sum(grad_w, axis=0)/N)
            grad_b_list.insert(0, np.sum(delta, axis=0)/N)
        if beta != 0:
            for i in range(self.nb_layers):
                grad_w_list[i] += beta * self.weights_list[i]
                grad_b_list[i] += beta * self.biases_list[i]
        #----    

        self.grad_w_list = grad_w_list
        self.grad_b_list = grad_b_list

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

        """
        Everytime we will reorder the list of inputs and use
        a stochastic process (a collection of batch_size random elements)
        to train our inputs. We will do it nb_batch times.
        And update the list of weights and the list of biases
        Parameters 
            ------------
        x_data : numpy.ndarray
            List of input values
        t_data : numpy.ndarray
            List of correct results. "1" means red_point and "0" means black_point
        epochs : 
        initialize_weights
        epsilon : 
        beta :
        print_cost    : Bool
        """
        if initialize_weights:
            self.init_weights()
            pass
        #your code here
        def SGD_update():
            self.weights_list[k] -= eta * self.grad_w_list[k]
            self.biases_list[k] -= eta * self.grad_b_list[k]

        def momentum_update():
            self.grad_w_averages[k] = gamma * self.grad_w_averages[k] + eta * self.grad_w_list[k]
            self.grad_b_averages[k] = gamma * self.grad_b_averages[k] + eta * self.grad_b_list[k]

            self.weights_list[k] -= self.grad_W_averages[k]
            self.biases_list[k] -= self.grad_b_averages[k]
            pass

        def adagrad_update():
            self.grad_w_averages[k] = np.square(self.grad_w_list[k])
            self.grad_b_averages[k] = np.square(self.grad_b_list[k])
            
            self.weights_list[k] -= eta * self.grad_w_list[k] / np.sqrt(self.grad_w_averages[k] + epsilon)
            self.biases_list[k] -= eta * self.grad_b_list[k] / np.sqrt(self.grad_b_averages[k] + epsilon)
            pass

        def RMSprop_update():
            self.grad_w_averages[k] = gamma * self.grad_w_averages[k] + (1 - gamma) * np.square(self.grad_w_list[k])
            self.grad_b_averages[k] = gamma * self.grad_b_averages[k] + (1 - gamma) * np.square(self.grad_b_list[k])
            
            self.weights_list[k] -= eta * self.grad_w_list[k] / np.sqrt(self.grad_w_averages[k] + epsilon)
            self.biases_list[k] -= eta * self.grad_b_list[k] / np.sqrt(self.grad_b_averages[k] + epsilon)
            pass

        def adadelta_update():
            self.grad_w_averages[k] = gamma * self.grad_w_averages[k] + (1 - gamma) * np.square(self.grad_w_list[k])
            self.grad_b_averages[k] = gamma * self.grad_b_averages[k] + (1 - gamma) * np.square(self.grad_b_list[k])
            
            self.delta_w2[k] = - self.grad_w_list[k] * np.sqrt(self.delta_w2[k] + epsilon) / np.sqrt(self.grad_w_averages[k] + epsilon)
            self.delta_b2[k] = - self.grad_b_list[k] * np.sqrt(self.delta_b2[k] + epsilon) / np.sqrt(self.grad_b_averages[k] + epsilon)

            self.weights_list[k] += self.delta_w2[k]
            self.biases_list[k] += self.delta_b2[k]
            pass

        def nesterov_update():
            pass

        def adam_update():
            self.grad_w_averages[k] = beta_1 / (1 - beta_1) * self.grad_w_averages[k] + self.grad_w_list[k]
            self.grad_b_averages[k] = beta_1 / (1 - beta_1) * self.grad_b_averages[k] + self.grad_b_list[k]
            self.gradsquare_w_averages[k] = beta_2 / (1 - beta_2) * self.grad_w_averages[k] + self.grad_w_list[k] * self.grad_w_list[k]
            self.gradsquare_b_averages[k] = beta_2 / (1 - beta_2) * self.grad_b_averages[k] + self.grad_b_list[k] * self.grad_b_list[k]
            
            self.weights_list[k] -= eta * self.grad_w_averages[k] / (np.sqrt(self.gradsquare_w_averages[k]) + epsilon)
            self.biases_list[k] -= eta * self.grad_b_averages[k] / (np.sqrt(self.gradsquare_b_averages[k]) + epsilon)
            pass
        #----

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
