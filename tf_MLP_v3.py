from __future__ import print_function, division

import tensorflow as tf
import numpy as np
import sys
from datetime import datetime

NOW = datetime.utcnow().strftime("%Y%m%d%H%M%S")
ROOT_LOGDIR = 'tf_logs'
LOG_DIR = "{}/run-{}".format(ROOT_LOGDIR, NOW)


class MLP(object):

    def __init__(self, K_list, activation_functions):
        tf.reset_default_graph()
        self.file_writer = None
        self.K_list = K_list
        self.activation_functions = activation_functions
        self.activations_dict = {'relu': tf.nn.relu,
                                 'sigmoid': tf.nn.sigmoid,
                                 'tanh': tf.nn.tanh,
                                 'identity': tf.identity}
        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}
        self.create_net()

    def fc_layer(self, unit, dim, activation):
        h = self.activations_dict[activation]
        with tf.name_scope('fc_layer'):
            w_shape = (int(unit.get_shape()[1]), dim)
            w = tf.Variable(tf.truncated_normal(w_shape), name='weights')
            b = tf.Variable(tf.zeros(dim), name='bias')
            a = tf.add(tf.matmul(unit, w), b, name='activation') #arregladlo

            z = h(a, name='unit')
            return z

    def create_net(self):
        K_list = self.K_list
        D = K_list[0]
        K = K_list[-1]

        self.x = tf.placeholder(tf.float32, shape=(None, D), name='x')
        self.y_ = tf.placeholder(tf.float32, shape=(None, K), name='y_')

        Z = self.fc_layer(self.x, K_list[1], self.activation_functions[0])
        for k_dim, activation in zip(K_list[2: -1],
                                     self.activation_functions[1:-1]):
            Z = self.fc_layer(Z, k_dim, activation)
        self.y = self.fc_layer(Z, K, 'identity')

        with tf.name_scope('loss'):
            loss_fn = self.loss_dict[self.activation_functions[-1]]
            self.loss = tf.reduce_mean(loss_fn(logits=self.y,
                                              labels=self.y_),
                                       name='loss')
            optimizer = tf.train.AdamOptimizer(0.01, name='optimizer')
            self.train_step = optimizer.minimize(self.loss, name='train_step')

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        file_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())

    def train(self, x_data, t_data,
              nb_epochs=100, batch_size=10):

        nb_data = x_data.shape[0]
        index_list = np.arange(nb_data)
        nb_batches = nb_data // batch_size

        with tf.Session() as sess:
            sess.run(self.init)
            for epoch in range(nb_epochs):
                np.random.shuffle(index_list)
                for batch in range(nb_batches):
                    batch_indices = index_list[batch * batch_size:
                                               (batch + 1) * batch_size]
                    x_batch = x_data[batch_indices, :]
                    t_batch = t_data[batch_indices, :]
                    sess.run(self.train_step,
                             feed_dict={self.x: x_batch,
                                        self.y_: t_batch})
                cost = sess.run(self.loss, feed_dict={self.x: x_data,
                                                      self.y_: t_data})
                sys.stdout.write('cost=%f %d\r' % (cost, epoch))
                sys.stdout.flush()
            self.saver.save(sess, "./MLP.ckpt")

    def predict(self, x_test):
        with tf.Session() as sess:
            self.saver.restore(sess, "./MLP.ckpt")
            pred = tf.nn.softmax(self.y)
            y_pred = sess.run(pred, feed_dict={self.x: x_test})
        return y_pred
