def NetConstructor(object):
    def __init__(self, layers):
        '''
        layers es una lista de diccionarios que describirán las sucesivas capas de la red.
        layers = [layer_0, layer_2, layer_3, ..., layer_m]
        layer_0 contendrá solamente la dimensión de los datos de entrada, que será una tupla o un número:
        Por ejemplo, layer_0 = {'dim': (224, 224, 3)} en el caso de que los 
        datos de entrada sean imágenes de dimensión 224 x 224 y 3 canales de color o
        layer_0 = {'dim': 784} en el caso de que sean vectores que representen imágenes de MNIST.
        En las restantes capas, la estructura será la siguiente:
        (se indican los parámetros mínimos que deberán estar implementados, se
         pueden añadir otros si se desea.)
        layer_k = {'type': layer_type, # tipo de capa: 'fc', 'conv', 'maxpool', 'dropout', 'LRN', ...
                   'dim': (dim_0, ..., dim_L) # dimensiones de la salida
                                              # de la capa (en su caso)
                   'kernel_size': size # por ejemplo, (3, 3) en una máscara convolucional 3 x 3
                   'stride': stride # por ejemplo, (1, 1) si se hace stride 1 horizontal y 1 vertical
                   'init': init_method # método de inicialización de pesos y biases, por ejemplo
                                       # ('truncated_normal', stddev, 'zeros'), 'xavier' o 'he'
                   'padding': padding # 'SAME', 'VALID'
                   'activation': activation, # función de activación, 
                                             # 'sigmoid', 'tanh', 'relu', 'identity', ...
                   'prob': probability, # float, probabilidad usada en dropout
                   'LRN_params': (k, alpha, beta, r)}
        '''
        self.layer_dict1d = {'fc': fc_layer1d,
                             'conv': tf.layers.conv1d,
                             'maxpool': tf.layers.max_pooling1d,
                             'dropout': tf.layers.dropout,
                             'LRN': }
        self.layer_dict2d = {'fc': fc_layer2d,
                             'conv': tf.layers.conv2d,
                             'maxpool': tf.layers.max_pooling2d,
                             'dropout': tf.layers.dropout,
                             'LRN': }
        self.layer_dict3d = {'fc': fc_layer3d,
                             'conv': tf.layers.conv3d,
                             'maxpool': tf.layers.max_pooling3d,
                             'dropout': tf.layers.dropout,
                             'LRN': }
        self.init_dict = {'truncated_normal':
                          'xavier': tf.layers.xavier_initializer,
                          'he': }        
        self.activations_dict = {'relu': tf.nn.relu,
                                 'sigmoid': tf.nn.sigmoid,
                                 'tanh': tf.nn.tanh,
                                 'identity': tf.identity}
        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}
        self.optimizers_dict = {'SGD': tf.train.GradientDescentOptimizer,
                                'adam': tf.train.AdamOptimizer,
                                'adagrad': tf.train.AdagradOptimizer,
                                'RMS_prop': tf.train.RMSPropOptimizer,
                                'adadelta': tf.train.AdadeltaOptimizer,
                                'momentum': tf.train.MomentumOptimizer,
                                'nesterov': tf.train.MomentumOptimizer} #Poner use_nesterov = True
        self.layers = layers
        self.create_net()

    def fc_layer1d(self, unit, dim, activation):
        h = self.activations_dict[activation]
        with tf.name_scope('fc_layer'):
            w_shape = (int(unit.get_shape()[1]), dim)
            w = tf.Variable(tf.truncated_normal(w_shape), name='weights')
            b = tf.Variable(tf.zeros(dim), name='bias')
            a = tf.add(tf.matmul(unit, w), b, name='activation')

            z = h(a, name='unit')
            return z
    
    def create_net(self):
        self.x = tf.placeholder(tf.float32, shape=(None, self.layers[0]), name='x')
        self.y_ = tf.placeholder(tf.float32, shape=(None, self.layers[-1]), name='y_')

        Z = self.fc_layer(self.x, self.layers[1], self.activation_functions[0])
        for k_dim, activation in zip(self.layers[2: -1],
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
    
    def train(self, x_train, t_train,
              nb_epochs=1000,
              batch_size=10,
              method=('adam', method_params)
              seed=seed_nb):
        '''
        El método train entrenará la red, recibiendo los datos de entrenamiento,
        número de epochs, tamaño del batch, una semilla opcional para el
        generador de números aleatorios y el método de entrenamiento:

        method = (str, params),

        el primer elemento describe el método de optimización,
        por ejemplo, 'SGD', 'nesterov', 'momentum', 'adagrad', 'adadelta', 'RMSprop'.
        El segundo elemento es un diccionario de parámetros adaptados al método,
        siguiendo la notación de la práctica anterior. Por ejemplo,
        method = ('SGD', {'eta': 0.1}) describirá un descenso de gradiente estocástico
        con learning rate = 0.1
        '''
        pass

    def predict(self, x_test):
        pass


if __name__ == "__main":
    layer_0 = {'dim': 30}
    layer_1 = {'type': 'fc', 'dim': 50, 'activation': 'sigmoid'}
    layer_2 = {'type': 'fc', 'dim': 10, 'activation': 'sigmoid'}
    layer_list = [layer_0, layer_1, layer_2]
    net = NetConstructor(layer_list);
   

 

'''


El método predict recibirá datos de test y devolverá la predicción de la red, una vez entrenada.
'''
