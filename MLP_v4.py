#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
authors: Xi Chen, Eric García de Ceca, Jaime Mendizábal Roche

This module is an advanced version of our MLP. In this module, we have
more kinds of optimization's methods, like "SDG", "momentum", "adagrad",
"RMS_prop", "adadelta", "nesterov" and "adam".
"""
from __future__ import division, print_function

import sys
import numpy as np
import tensorflow as tf

class MLP(object):
    """
    In the class MLP, we have those optimization's methods we have learnd.
    """
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
        """
        Definition of function sigmoid
        Args:
        z (real, vector): the input value of the function
        Returns:
        1 / (1 + np.exp(-z)) (real >0, <1)
        """
        # ->
        return 1 / (1 + np.exp(-z))
        # <-

    @staticmethod
    def dsigmoid(z):
        """
        Definition of function sigmoid's derivative
        Args:
        z (real, vector): the input value of the function.
        Returns:
        MLP.sigmoid(z)*(1 - MLP.sigmoid(z)) (real) :MLP.sigmoid(z) = 1 / (1 + np.exp(-z))
        """
        # ->
        return MLP.sigmoid(z)*(1 - MLP.sigmoid(z))
        # <-

    @staticmethod
    def dtanh(z):
        """
        Definition of function tanh's derivative
        Args:
        z (real, vector): the input value of the function.
        Returns:
        1 - np.tanh(z)**2 (real) : np.tanh(z) is the tanh function in numpy.
        """
        # ->
        return 1 - np.tanh(z)**2
        # <-

    @staticmethod
    def relu(z):
        """
        Definition of function Rectifier
        Args:
        z (real, vector): the input value of the function
        Returns:
        ret (real, vector): based on a same vector as z,it swaps those
        negative values with 0
        """
        # ->
        ret = np.copy(z)
        ret[z < 0] = 0
        return ret
        # <-

    @staticmethod
    def drelu(z):
        """
        Definition of function Rectifier's derivative
        Args:
        z (real, vector): the input value of the function.
        Returns:
        ret (real, vector): based on a same vector as z, it swaps those
        negative values with 0 and those no-negative values with 1
        """
        # ->
        ret = np.copy(z)
        ret[z < 0] = 0
        ret[z >= 0] = 1
        return ret
        # <-

    @staticmethod
    def identity(z):
        """
        Definition of function identity
        Args:
        z (real, vector): the input value of the function
        Returns:
        z (real, vector): the output values is as same as the input one
        """
        # ->
        return z
        # <-

    @staticmethod
    def didentity(z):
        """
        Definition of function didentity's derivative
        Args:
        z (real, vector): the input value of the function
        Returns:
        np.ones(z.shape()): a vector which is full of 1s and has the z's length
        """
        # ->
        return np.ones(z.shape())
        # <-

    @staticmethod
    def softmax(z):
        """
        Definition of function softmax
        Args:
        z (real, vector): the input value of the function
        Returns:
        exps / exps_sums[:, np.newaxis]: softmax of the input vector z
        """
        # ->
        exps = np.exp(z)
        exps_sums = np.sum(exps, axis=1)
        return exps / exps_sums[:, np.newaxis]
        # <-

    @staticmethod
    def binary_cross_entropy(y, t_data):
        """
        This is the function what computes the binary_cross_entropy value, with
        a vector of datas and the output values.
        """
        # ->
        return -np.sum(t_data * np.log(y) + (1 - t_data) * np.log(1 - y))
        # <-

    @staticmethod
    def softmax_cross_entropy(y, t_data):
        """
        This is the function what computes the softmax_cross_entropy value,
        with a vector of datas and the output values.
        """
        # ->
        return -np.sum(t_data * np.log(y))
        # <-

    @staticmethod
    def cost_L2(y, t_data):
        """
        This is the function what computes the cost, with a vector of datas and
        the output values.
        """
        # ->
        return np.sum((y-t_data)**2) / 2
        # <-

    def init_weights(self):
        """
        This function is used to initial weights_list (weights), grad_b_list
        (biases values), grad_w_averages(update vectorof weights in momentum),
        grad_b_averages(update vector of biases in momentum),
        gradsquare_w_averages(Gt of weights in adagrad) and
        gradsquare_b_averages(Gt of biases in adagrad).
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

    del fc_layer(self, z, dim, activ_funct):
        W_shape = (int(z.get_shape()[1]), dim)
        W-init = tf.truncated_normal(W_shape)
        W = tf.Variable(W-init, name="weights")
        b = tf.Variable(0.0, name="bias")
        a = tf.matmul(z,W) + b
        z = activ_funct(a)
        return z
        
    def get_activations_and_units(self, x, parameters=None):
        """
        In this function, we have nearly the same as previous MLP, but now with
        an extra parameters. The function is used to get activations and units.
        Args:
        x (real, vector): input value. data for train
        parameters (real, vector): input value. If it's None, the function will
                                   use the original lists of weights and biases
                                   to get activations and units. If not, it
                                   will use parameters as lists of weights and
                                   biases
        """
        # ->
        if parameters is None:
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
        """
        In this function, we have nearly the same as previous MLP, but now with
        an extra parameters.The function is used to get gradients.
        Args:
        x (real, vector): input value. data for train
        t (real, vector): List of correct results.
                        "1" means red_point and "0" means black_point
        parameters (real, vector): input value. If it's None, the function will
                                   use the original lists of weights and biases
                                   to get activations and units. If not, it
                                   will use parameters as lists of weights and
                                   biases
        """
        # ->
        if parameters is None:
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
        """
        It is the principal function of MLP, which trains the input datas with
        one of the optimization's methods. After trainning datas with this
        function, we will have the results and enough datas to print them

        Args:
        x_data (real, vector): input value. data for train
        t_data (real, vector): List of correct results.
                        "1" means red_point and "0" means black_point
        epochs: number of the epochs of train
        batch_size: size of the data to train in every epoch
        eta (real): learning rate
        beta (real): a value used to compute the gradients. It is 0 by default
        gamma (real): fraction of the update vector(momentum term). gamma is
                        usually 0.9.
        beta_1 (real): fraction of estimate of the first moment (the mean) of
                        the gradients
        beta_2 (real): fraction of estimate of  the second moment
                        (the uncentered variance) of the gradients
        epsilon (real): a smoothing term that avoids division by zero
                        (usually on the order of 1e−8)
        method (string, dictionary): name of method. It will be used to choose
                                    the optimization method
        print_cost (bool): boolean value to print or no the cost.

        """

        if initialize_weights:
            self.init_weights()
        """
        For the block "for" of the function train, we need a update
        of parameters(weights, biases, etc) for every optimization
        method using the form of Sebastian's website
        """
        def SGD_update():
            """
            Here we will update the values of weights and biases. It is the
            same thing we have done in our MLP
            """
            # ->
            self.weights_list[k] -= eta * self.grad_w_list[k]
            self.biases_list[k] -= eta * self.grad_b_list[k]
            # <-

        def momentum_update():
            """
            We have Vt=γVt−1+η∇θJ(θ) and θ=θ−Vt for "momentum".
            Vt is the update vector. ∇θJ(θ) is our lists of weights and
            biases. We use the 1st form to compute Vt.
            Then, the 2nd one to have our lists of weights and biases(θ).
            """
            # ->
            self.grad_w_averages[k] *= gamma
            self.grad_w_averages[k] += eta * self.grad_w_list[k]
            self.grad_b_averages[k] *= gamma
            self.grad_b_averages[k] += eta * self.grad_b_list[k]

            self.weights_list[k] -= self.grad_w_averages[k]
            self.biases_list[k] -= self.grad_b_averages[k]
            # <-

        def adagrad_update():
            """
            Adagrad is an algorithm for gradient-based optimization that
            does just this: It adapts the learning rate to the parameters,
            performing larger updates for infrequent and smaller updates
            for frequent parameters.Gt∈Rd×d here is a diagonal matrix
            where each diagonal element i,i is the sum of the squares of
            the gradients w.r.t. θi up to time step t, while eps(epsilon)
            is a smoothing term that avoids division by zero. And gt is
            the vector of the gradient of the objective function w.r.t. to
            the parameter θ at time step t. gt = ∇θJ(θ)
            We have these forms:
            Gt = Gt-1 + (gt)²
            θt+1 = θt − (η / √(Gt + eps)) * gt.
            η / √(Gt + eps) is named as new_eta in our function.
            """
            # ->
            self.gradsquare_w_averages[k] += self.grad_w_list[k]**2
            self.gradsquare_b_averages[k] += self.grad_b_list[k]**2

            new_eta = eta / np.sqrt(self.gradsquare_w_averages[k] + epsilon)
            self.weights_list[k] -= new_eta * self.grad_w_list[k]
            new_eta = eta / np.sqrt(self.gradsquare_b_averages[k] + epsilon)
            self.biases_list[k] -= new_eta * self.grad_b_list[k]
            # <-

        def RMSprop_update():
            """
            RMSprop is an unpublished, adaptive learning rate method
            proposed by Geoff Hinton. It is identical to the first update
            vector of Adadelta
            """
            # ->
            self.gradsquare_w_averages[k] *= gamma
            self.gradsquare_w_averages[k] += \
                (1 - gamma) * self.grad_w_list[k]**2
            self.gradsquare_b_averages[k] *= gamma
            self.gradsquare_b_averages[k] += \
                (1 - gamma) * self.grad_b_list[k]**2

            new_eta = eta / np.sqrt(self.gradsquare_w_averages[k] + epsilon)
            self.weights_list[k] -= new_eta * self.grad_w_list[k]
            new_eta = eta / np.sqrt(self.gradsquare_b_averages[k] + epsilon)
            self.biases_list[k] -= new_eta * self.grad_b_list[k]
            # <-

        def adadelta_update():
            """
            Adadelta is an extension of Adagrad that seeks to reduce its
            aggressive, monotonically decreasing learning rate. Instead of
            accumulating all past squared gradients, Adadelta restricts the
            window of accumulated past gradients to some fixed size w.
            θt+1 = θt + Δθt
            Δθt = −(RMS[Δθ]t−1 / RMS[g]t) * gt  (RMS: root mean squared)
            RMS[Δθ]t=√(E[Δθ2]t + eps)  (eps: epsilon)
            E[Δθ²]t=γ*E[Δθ²]t−1 + (1−γ)*Δθ²t
            """
            # ->
            self.gradsquare_w_averages[k] *= gamma
            self.gradsquare_w_averages[k] += \
                (1 - gamma) * self.grad_w_list[k]**2
            self.gradsquare_b_averages[k] *= gamma
            self.gradsquare_b_averages[k] += \
                (1 - gamma) * self.grad_b_list[k]**2

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
            """
            Nesterov accelerated gradient (NAG) is a way to give our momentum
            term this kind of prescience. Computing θ − γ*Vt−1 thus gives us
            an approximation of the next position of the parameters, a rough
            idea where our parameters are going to be. So we can compute ∇θJ
            with θ − γ*Vt−1, which will give us a better approximation:
            Vt = γ*Vt−1 + η*∇θJ(θ − γ*vt−1)
            θ = θ − Vt
            """
            # ->
            new_weights = []
            new_biases = []

            for k in range(self.nb_layers):
                self.grad_w_averages[k] *= gamma
                self.grad_b_averages[k] *= gamma

                new_weights.append(self.weights_list[k]
                                   - self.grad_w_averages[k])
                new_biases.append(self.biases_list[k]
                                  - self.grad_b_averages[k])

            params = (new_weights, new_biases)
            self.get_gradients(x_batch, t_batch, beta, parameters=params)

            for k in range(self.nb_layers):
                self.grad_w_averages[k] += eta * self.grad_w_list[k]
                self.grad_b_averages[k] += eta * self.grad_b_list[k]

                self.weights_list[k] -= self.grad_w_averages[k]
                self.biases_list[k] -= self.grad_b_averages[k]
            # <-

        def adam_update():
            """
            Adaptive Moment Estimation (Adam) is another method that
            computes adaptive learning rates for each parameter. We need to
            store an exponentially decaying average of past squared
            gradients vt like Adadelta and RMSprop, and an exponentially
            decaying average of past gradients mt, similar to momentum:
            mt = β1*mt−1 + (1−β1)*gt    (1st moment)
            vt = β2*vt−1 + (1−β2)*gt²   (2nd moment)
            And we need to counteract these biases of mt and vt by
            computing bias-corrected mt and vt:
            m = m / (1−β1**t)   v = v / (1−β2**t)  In our function, they
            are named as m_w, m_b, v_w, v_b.
            """
            # ->
            self.grad_w_averages[k] *= beta_1
            self.grad_w_averages[k] += (1 - beta_1) * self.grad_w_list[k]
            self.grad_b_averages[k] *= beta_1
            self.grad_b_averages[k] += (1 - beta_1) * self.grad_b_list[k]

            self.gradsquare_w_averages[k] *= beta_2
            self.gradsquare_w_averages[k] += \
                (1 - beta_2) * self.grad_w_list[k]**2
            self.gradsquare_b_averages[k] *= beta_2
            self.gradsquare_b_averages[k] += \
                (1 - beta_2) * self.grad_b_list[k]**2

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
                    cost = MLP.binary_cross_entropy(self.y, t_data)
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
