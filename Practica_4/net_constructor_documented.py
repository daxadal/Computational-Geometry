"""
authors: Xi Chen, Eric García de Ceca, Jaime Mendizábal Roche
This module is an advanced version of our MLP using tensorflow.
In this module, we have optimization's methods, like "SDG",
"momentum", "adagrad","RMS_prop", "adadelta", "nesterov" and "adam".
"""
from __future__ import print_function, division

import sys
from datetime import datetime
import tensorflow as tf
import numpy as np


NOW = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = 'tf_logs'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)

class NetConstructor(object):
    """
    In the class NetConstructor, we construct a Convolutional Neural Network.
    """
    def __init__(self, layers):
        tf.reset_default_graph()

        if type(layers[0]['dim']) is int:
            layers[0]['dim'] = (layers[0]['dim'],)
        self.layers = layers

        self.activations_dict = {'relu': tf.nn.relu,
                                 'sigmoid': tf.nn.sigmoid,
                                 'tanh': tf.nn.tanh,
                                 'identity': tf.identity,
                                 'softmax': tf.nn.softmax}

        self.create_net()

    def create_net(self):
        """
        Create the Convolutional Neural Network, those basic values and layers.
        """
        def init_tn(shape, w_or_b):
            """
            Outputs random values from a truncated normal distribution.
            """
            return tf.truncated_normal(shape, stddev=layer['stddev_' + w_or_b])

        def init_zeros(shape, w_or_b=None):
            """
            Create a zero matrix.
            """
            return tf.zeros(shape)

        def reduce_mul(vector):
            """
            Compute the product of all numbers in a vector.
            """
            ret = 1
            for i in range(len(vector)):
                ret *= vector[i]
            return ret

        def fc_layer():
            """
            The Fully Connected layer is a traditional Multi Layer Perceptron that uses
            an activation function in the output layer.
            The term “Fully Connected” implies that every neuron in the previous layer is
            connected to every neuron on the next layer. Their activations can hence be computed
            with a matrix multiplication followed by a bias offset. The purpose of the Fully
            Connected layer is to use these features for classifying the input image into
            various classes based on the training dataset.
            1st, we will get the values of dim, activation, weights and biases from layer.
            2nd, we will reshape the unit to a vector and the weights to a vector (length dim).
            3rd, we use the Variable of tensorflow to get weights and bias.
            Finally, with the famous function: wX+b, we can get the activ and h.
            """
            with tf.name_scope('fc_layer'):
                dim = layer['dim']

                h = self.activations_dict[layer['activation']]
                init_w = init_dict[layer['init_w']]
                init_b = init_dict[layer['init_b']]

                unit_dim = [-1] + [reduce_mul(unit.get_shape().as_list()[1:])]
                reshaped_unit = tf.reshape(unit, unit_dim)
                weights_shape = (unit_dim[1], dim)

                weights = tf.Variable(init_w(weights_shape, w_or_b='w'), name='weights')
                bias = tf.Variable(init_b(dim, w_or_b='b'), name='bias')

                activ = tf.add(tf.matmul(reshaped_unit, weights), bias, name='activation')
                return h(activ, name='unit')

        def conv_layer():
            """
            Convolutional Layer
            The Conv layer is the core building block of a Convolutional Network
            that does most of the computational heavy lifting.
            The process is : do the product of part of unit(filter_size) and filter,
            then compute the sum and add the biases.
            """
            with tf.name_scope('conv_layer'):
                h = self.activations_dict[layer['activation']]
                init_w = init_dict[layer['init_w']]
                init_b = init_dict[layer['init_b']]

                filter_size = layer['k_size']+(int(unit.get_shape()[3]), layer['channels'])

                filter = tf.Variable(init_w(filter_size, w_or_b='w'), name='filter')

                aux = tf.nn.conv2d(input=unit,
                                   filter=filter,
                                   strides=(1,) + layer['strides'] + (1,),
                                   padding=layer['padding'],
                                   name='activation_before_bias')

                biases = tf.Variable(init_b(layer['channels'], w_or_b='b'), name='biases')

                activ = tf.nn.bias_add(aux, biases, name='activation')
                return h(activ, name='unit')

        def maxpool_layer():
            """
            Performs the max pooling(take the biggest value)on the input.

            value: A Tensor. Unit
            ksize: A list of size of the window for each dimension of the input tensor.
            strides: A list of the stride of the sliding window for each dimension of
                     the input tensor.
            padding: A string, either 'VALID' or 'SAME'. 'maxpool_layer'
            name: Optional name for the operation.
            """
            with tf.name_scope('maxpool_layer'):
                return tf.nn.max_pool(value=unit,
                                      ksize=(1,) + layer['k_size'] + (1,),
                                      strides=(1,) + layer['strides'] + (1,),
                                      padding=layer['padding'],
                                      name='maxpool_layer')

        def dropout_layer():
            """
            Computes dropout.
            With probability keep_prob, outputs the input element scaled up by keep_prob,
            otherwise outputs 0. The scaling is so that the expected sum is unchanged.
            By default, each element is kept or dropped independently.

            x: A tensor.
            keep_prob: A scalar Tensor with the same type as x.
                       The probability that each element is kept.
            name: A name for this operation (optional).

            """
            with tf.name_scope('dropout_layer'):
                return tf.nn.dropout(x=unit,
                                     keep_prob=layer['prob'],
                                     name='dropout_layer')

        def LRN_layer():
            """
            Local Response Normalization defined by tensorflow.The 4-D input tensor
            is treated as a 3-D array of 1-D vectors (along the last dimension),
            and each vector is normalized independently. Within a given vector,
            each component is divided by the weighted, squared sum of inputs
            within depth_radius. In detail,
            sqr_sum[a, b, c, d] = sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
            output = input / (bias + alpha * sqr_sum) ** beta

            These values will be given by our layer:
            input: A Tensor. Our unit.
            depth_radius: An optional int.
            bias: An optional float. An offset (usually positive to avoid dividing by 0).
            alpha: An optional float. A scale factor, usually positive.
            beta: An optional float. Defaults to 0.5. An exponent.
            name: A name for the operation (optional). We use here 'LRN_layer'
            """
            with tf.name_scope('LRN_layer'):
                return tf.nn.local_response_normalization(
                    input=unit,
                    bias=layer['k'],
                    alpha=layer['alpha'],
                    beta=layer['beta'],
                    depth_radius=layer['r'],
                    name='LRN_layer')

        def BN_layer():
            """
            Batch normalization: Normalizes a tensor by mean and variance,
            and applies (optionally) a scale to it, as well as an offset:
            (scale * (input - mean) / variance) + offset
            """
            with tf.name_scope('BN_layer'):
                mean, variance = tf.nn.moments(unit, [0], keep_dims=True)
                offset = tf.Variable(tf.zeros(mean.get_shape()), name='offset')
                scale = tf.Variable(tf.ones(variance.get_shape()), name='scale')
                return tf.nn.batch_normalization(
                    x=unit,
                    mean=mean,
                    variance=variance,
                    offset=offset,
                    scale=scale,
                    variance_epsilon=1e-8,
                    name='BN_layer')

        layer_dict = {'fc': fc_layer,
                      'conv': conv_layer,
                      'maxpool': maxpool_layer,
                      'dropout': dropout_layer,
                      'LRN': LRN_layer,
                      'BN': BN_layer}
        init_dict = {'truncated_normal': init_tn,
                     'zeros': init_zeros}
        loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                     'identity': tf.nn.l2_loss,
                     'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}

        self.x = tf.placeholder(tf.float32, shape=(None,)+self.layers[0]['dim'], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=(None, self.layers[-1]['dim']), name='y_')
        unit = self.x;

        for layer in self.layers[1:-1]:
            layer_funct = layer_dict[layer['type']]
            unit = layer_funct()

        layer = self.layers[-1]

        layer_funct = layer_dict[layer['type']]

        self.last_activation = layer['activation']
        layer['activation'] = 'identity'

        self.logits = layer_funct()

        layer['activation'] = self.last_activation

        with tf.name_scope('loss'):
            loss_funct = loss_dict[self.last_activation]
            self.loss = tf.reduce_mean(loss_funct(logits=self.logits,
                                                  labels=self.y_),
                                       name='loss')

        self.saver = tf.train.Saver()
        file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())

    def train(self, x_train, t_train,
              nb_epochs=1000,
              batch_size=10,
              method=('SGD', {'eta': 0.1}),
              seed=tf.set_random_seed(1),
              use_validation=False,
              x_val=None,
              t_val=None,
              show_cost=True,
              load=False):
        """
        It is the principal function of MLP, which trains the input datas with
        our Convolutional Neural Network. After trainning datas with this
        function, we will have the results and enough datas to print them
        Args:
        x_train: input value. data for train
        t_train: List of correct results.
                 "1" means red_point and "0" means black_point
        nb_epochs: number of the epochs of train
        batch_size: size of the data to train in every epoch
        method: name of method and the parameters, like eta (learning rate),
                beta (a value used to compute the gradients), gamma (fraction of the update vector
                or momentum term), beta_1 (fraction of estimate of the first moment (the mean) of
                the gradients), beta_2 (fraction of estimate of  the second moment), epsilon (a
                smoothing term that avoids division by zero), etc.
        seed: A Python integer. Used to create random seeds.
        use_validation: boolean value to run or no the validation function
        show_cost: boolean value to print or no the cost.
        """
        def SGD_adapter(params, name):
            """
            We use the function defined by tensorflow.
            Args:
            params: It is a tuple which has the values of learning_rate, beta1, beta2,
                    epsilon, decay, rho and momentum. We only use learning_rate here, which
                    is a Tensor or a floating point value.(eta)
            name: Optional name for the operations created when applying gradients.
                  Defaults to "GradientDescent".
            Returns:
            A new gradient descent optimizer
            """
            return tf.train.GradientDescentOptimizer(
                learning_rate=params['eta'],
                name=name)

        def adam_adapter(params, name):
            """
            We use the function defined by tensorflow.
            Args:
            params: It is a tuple which has the values of learning_rate, beta1, beta2,
                    epsilon, decay, rho and momentum. We use:
                    learning_rate: A Tensor or a floating point value. The learning rate.
                    beta1: A float value or a constant float tensor.
                           The exponential decay rate for the 1st moment estimates
                    beta2: A float value or a constant float tensor.
                           The exponential decay rate for the 2nd moment estimates.
                    epsilon: A small constant for numerical stability.
            name: Optional name for the operations created when applying gradients.
                  Defaults to "Adam".
            Returns:
            A new Adam optimizer
            """
            return tf.train.AdamOptimizer(
                learning_rate=params['eta'],
                beta1=params['beta_1'],
                beta2=params['beta_2'],
                epsilon=params['epsilon'],
                name=name)

        def adagrad_adapter(params, name):
            """
            We use the function defined by tensorflow.
            Args:
            params: It is a tuple which has the values of learning_rate, beta1, beta2,
                    epsilon, decay, rho and momentum. We use:
                    learning_rate: A Tensor or a floating point value. The learning rate.
            name: Optional name prefix for the operations created when applying gradients.
                  Defaults to "Adagrad".
            Returns:
            A new Adagrad optimizer.
            """
            return tf.train.AdagradOptimizer(
                learning_rate=params['eta'],
                name=name)

        def RMS_adapter(params, name):
            """
            We use the function defined by tensorflow.
            Args:
            params: It is a tuple which has the values of learning_rate, beta1, beta2,
                    epsilon, decay, rho and momentum. We use:
                    learning_rate: A Tensor or a floating point value. The learning rate.
                    decay: Discounting factor for the history/coming gradient
                    epsilon: Small value to avoid zero denominator.
            name: Optional name prefix for the operations created when applying gradients.
                  Defaults to "RMSProp".
            Returns:
            A new RMSProp optimizer.
            """
            return tf.train.RMSPropOptimizer(
                learning_rate=params['eta'],
                decay=params['gamma'],
                epsilon=params['epsilon'],
                name=name)

        def adadelta_adapter(params, name):
            """
            We use the function defined by tensorflow.
            Args:
            params: It is a tuple which has the values of learning_rate, beta1, beta2,
                    epsilon, decay, rho and momentum. We use:
                    learning_rate: A Tensor or a floating point value. The learning rate.
                    rho: A Tensor or a floating point value. The decay rate.
                    epsilon: A Tensor or a floating point value.
                             A constant epsilon used to better conditioning the grad update.
            name: Optional name prefix for the operations created when applying gradients.
                  Defaults to "Adadelta".
            Returns:
            A new Adadelta optimizer.
            """
            return tf.train.AdadeltaOptimizer(
                learning_rate=params['eta'],
                rho=params['gamma'],
                epsilon=params['epsilon'],
                name=name)

        def momentum_adapter(params, name):
            """
            We use the function defined by tensorflow.
            Args:
            params: It is a tuple which has the values of learning_rate, beta1, beta2,
                    epsilon, decay, rho and momentum. We use:
                    learning_rate: A Tensor or a floating point value. The learning rate.
                    momentum: A Tensor or a floating point value. The momentum.
            name: Optional name prefix for the operations created when applying gradients.
                  Defaults to "Momentum".
            Returns:
            A new Momentum optimizer
            """
            return tf.train.MomentumOptimizer(
                learning_rate=params['eta'],
                momentum=params['gamma'],
                name=name)

        def nesterov_adapter(params, name):
            """
            We use the function defined by tensorflow. We need 4 arguments for this function.
            Args:
            params: It is a tuple which has the values of learning_rate, beta1, beta2,
                    epsilon, decay, rho and momentum. We use:
                    learning_rate: A Tensor or a floating point value. The learning rate.
                    momentum: A Tensor or a floating point value. The momentum.
                    use_nesterov: If True use Nesterov Momentum
            name: Optional name prefix for the operations created when applying gradients.
                  Defaults to "Momentum".
            Returns:
            A new Nesterov optimizer
            """
            return tf.train.MomentumOptimizer(
                learning_rate=params['eta'],
                momentum=params['gamma'],
                use_nesterov=True,
                name=name)

        optimizers_dict = {'SGD': SGD_adapter,
                           'adam': adam_adapter,
                           'adagrad': adagrad_adapter,
                           'RMS_prop': RMS_adapter,
                           'adadelta': adadelta_adapter,
                           'momentum': momentum_adapter,
                           'nesterov': nesterov_adapter}

        with tf.name_scope('train'):
            method_class = optimizers_dict[method[0]]
            optimizer = method_class(method[1], name='optimizer')

            self.train_step = optimizer.minimize(self.loss, name='train_step')

        self.init = tf.global_variables_initializer()

        nb_data = x_train.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = nb_data // batch_size

        with tf.Session() as sess:
            if load:
                sess.run(self.init)
                self.saver.restore(sess, "./MLP.ckpt")
            else:
                sess.run(self.init)

            for epoch in range(nb_epochs):
                np.random.shuffle(index_list)
                for batch in range(nb_batches):
                    batch_indices = index_list[batch * batch_size:
                                               (batch + 1) * batch_size]
                    x_batch = x_train[batch_indices, :]
                    t_batch = t_train[batch_indices, :]

                    sess.run(self.train_step,
                             feed_dict={self.x: x_batch,
                                        self.y_: t_batch})

                cost = 0.0
                if use_validation:
                    cost = sess.run(self.loss, feed_dict={self.x: x_val,
                                                          self.y_: t_val})
                else:
                    if show_cost:
                        cost = sess.run(self.loss, feed_dict={self.x: x_train,
                                                              self.y_: t_train})
                sys.stdout.write('cost=%f %d\r' % (cost, epoch))
                sys.stdout.flush()

            self.saver.save(sess, "./MLP.ckpt")


    def predict(self, x_test):
        with tf.Session() as sess:
            self.saver.restore(sess, "./MLP.ckpt")

            pred = self.activations_dict[self.last_activation](self.logits)
            y_pred = sess.run(pred, feed_dict={self.x: x_test})

        return y_pred


if __name__ == "__main__":
    nb_black = 15
    nb_red = 15
    nb_data = nb_black + nb_red
    x_data_black = np.random.randn(nb_black, 2) + np.array([0, 0])
    x_data_red = np.random.randn(nb_red, 2) + np.array([10, 10])
    x_data = np.vstack((x_data_black, x_data_red))
    t_data = np.asarray([0]*nb_black + [1]*nb_red).reshape(nb_data, 1)

    layer_0 = {'dim': 2}
    layer_1 = {'type': 'fc', 'dim': 50, 'activation': 'sigmoid', 'init': 'he'}
    layer_2 = {'type': 'fc', 'dim': 1, 'activation': 'sigmoid', 'init': 'xavier'}
    layer_list = [layer_0, layer_1, layer_2]
    net = NetConstructor(layer_list)
net.train(x_data, t_data)
