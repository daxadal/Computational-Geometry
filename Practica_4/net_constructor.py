from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import sys
from datetime import datetime

NOW = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = 'tf_logs'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)

class NetConstructor(object):
    def __init__(self, layers):
        tf.reset_default_graph()
        
        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}
        for i in range(len(layers)):
            if type(layers[i]['dim']) is int:
                layers[i]['dim'] = (layers[i]['dim'],)
            if not ('init' in layers[i]):
                layers[i]['init'] = 'truncated_normal'
        self.layers = layers
        self.create_net()
    
    def create_net(self):
        def index(dim):
            if type(dim) is tuple: index = len(dim)-1
            else: index = 0
            return index

        def reduce_mul(l):
            ret = 1
            for i in range(len(l)):
                ret *= l[i]
            return ret
        
        def fc_layer(unit, dim, params):
            '''
            if index(dim) == 2: newdim = dim[0] * dim[1] * dim[2]
            elif index(dim) == 1: newdim = dim[0] * dim[1]
            else: newdim = dim
            '''

            h = activations_dict[params['activation']]
            flat_dim = reduce_mul(dim)
            #final_dim = list((int(unit.get_shape()[0]),) + dim)
            unit_dim = [-1] + [reduce_mul(unit.get_shape().as_list()[1:])]
            final_dim = list((-1,)+dim)
            #
            init = self.init_dict[params['init']]
            #init = tf.truncated_normal(weights_shape)
            with tf.name_scope('fc_layer'):
                reshaped_unit = tf.reshape(unit, unit_dim)
                weights_shape = (unit_dim[1], flat_dim)
                #weights = tf.Variable(tf.truncated_normal(weights_shape), name='weights')
                weights = tf.Variable(init(weights_shape), name='weights')
                bias = tf.Variable(tf.zeros(flat_dim), name='bias')
                activ = tf.add(tf.matmul(reshaped_unit, weights), bias, name='activation')
                return h(tf.reshape(activ, final_dim), name='unit')
        
        def conv_layer(unit, dim, params):
            activation = activations_dict[params['activation']]
            #init = self.init_dict[params['init']]
            init = tf.contrib.layers.xavier_initializer()
            layer_list = [tf.layers.conv1d, tf.layers.conv2d, tf.layers.conv3d]
            #index(dim) o len(params['kernel_size'])
            conv_dim = len(params['kernel_size'])
            return layer_list[conv_dim](inputs=unit,
                                        filters=dim,
                                        kernel_size=params['kernel_size'],
                                        strides=params['stride'],
                                        padding=params['padding'],
                                        activation=activation,
                                        kernel_initializer=init,
                                        bias_initializer=init,
                                        name='conv_layer')
        
        def maxpool_layer(unit, dim, params):
            layer_list = [tf.layers.max_pooling1d, tf.layers.max_pooling2d, tf.layers.max_pooling3d]
            conv_dim = len(params['kernel_size'])
            return layer_list[conv_dim](inputs=unit,
                                    pool_size=params['kernel_size'],
                                    strides=params['stride'],
                                    padding=params['padding'],
                                    name='maxpool_layer')
                                    
        def dropout_layer(unit, dim, params):
            return tf.layers.dropout(inputs=unit,
                                     rate=params['prob'],
                                     name='dropout_layer')
        
        def LRN_layer(unit, dim, params):
            return tf.nn.local_response_normalization(
                              inputs=unit,
                              bias=params['LRN_params'][0],
                              alpha=params['LRN_params'][1],
                              beta=params['LRN_params'][2],
                              depth_radius=params['LRN_params'][3],
                              name='LRN_layer')
                              
        layer_dict = {'fc': fc_layer,
                      'conv': conv_layer,
                      'maxpool': maxpool_layer,
                      'dropout': dropout_layer,
                      'LRN': LRN_layer}
        activations_dict = {'relu': tf.nn.relu,
                            'sigmoid': tf.nn.sigmoid,
                            'tanh': tf.nn.tanh,
                            'identity': tf.identity,
                            'softmax': tf.nn.softmax}
        self.init_dict = {'truncated_normal': tf.truncated_normal_initializer(),
                          'xavier': tf.contrib.layers.xavier_initializer(),
                          'he': tf.contrib.keras.initializers.he_normal()} # or uniform?
        
                                
        self.x = tf.placeholder(tf.float32, shape=(None,)+self.layers[0]['dim'], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=(None,)+self.layers[-1]['dim'], name='y_')
        unit = self.x;
        for layer in self.layers[1:-1]:
            layer_type = layer_dict[layer['type']]
            unit = layer_type(unit, layer['dim'], layer)
            
        layer_type = layer_dict[self.layers[-1]['type']]
        self.last_activation = self.layers[-1]['activation']
        self.layers[-1]['activation'] = 'identity'
        self.logits = layer_type(unit, self.layers[-1]['dim'], self.layers[-1]) #la ultima capa debe tener activacion identity
    
    def train(self, x_train, t_train,
              nb_epochs=1000,
              batch_size=10,
              method=('SGD', {'eta': 0.1}),
              seed=tf.set_random_seed(1)):
        def SGD_adapter(params, name):
            return tf.train.GradientDescentOptimizer(
                learning_rate=params['eta'],
                name=name)
                
        def adam_adapter(params, name):
            return tf.train.AdamOptimizer(
                learning_rate=params['eta'],
                beta1=params['beta_1'],
                beta2=params['beta_2'],
                epsilon=params['eps'],
                name=name)
                
        def adagrad_adapter(params, name):
            return tf.train.AdagradOptimizer(
                learning_rate=params['eta'],
                #initial_accumulator_value=0.1,
                name=name)
                    
        def RMS_adapter(params, name):
            return tf.train.RMSPropOptimizer(
                learning_rate=params['eta'],
                decay=params['gamma'],
                epsilon=params['eps'],
                name=name)
                
        def adadelta_adapter(params, name):
            return tf.train.AdadeltaOptimizer(
                learning_rate=params['eta'],
                rho=params['gamma'],
                epsilon=params['eps'],
                name=name)
                
        def momentum_adapter(params, name):
            return tf.train.MomentumOptimizer(
                learning_rate=params['eta'],
                momentum=params['gamma'],
                name=name)
                
        def nesterov_adapter(params, name):
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
                           
        with tf.name_scope('loss'):
            method_class = optimizers_dict[method[0]]
            loss_fn = self.loss_dict[self.last_activation]
            self.loss = tf.reduce_mean(loss_fn(logits=self.logits,
                                              labels=self.y_),
                                       name='loss')
            optimizer = method_class(method[1], name='optimizer')
            self.train_step = optimizer.minimize(self.loss, name='train_step')

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
        
        nb_data = x_train.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = nb_data // batch_size

        with tf.Session() as sess:
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
                cost = sess.run(self.loss, feed_dict={self.x: x_train,
                                                      self.y_: t_train})
                sys.stdout.write('cost=%f %d\r' % (cost, epoch))
                sys.stdout.flush()
            self.saver.save(sess, "./MLP.ckpt")
        

    def predict(self, x_test):
        with tf.Session() as sess:
            self.saver.restore(sess, "./MLP.ckpt")
            pred = self.last_activation(self.logits)
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
