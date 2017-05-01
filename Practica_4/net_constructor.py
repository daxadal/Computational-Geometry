def NetConstructor(object):
    def __init__(self, layers):
        '''
        layers es una lista de diccionarios que describirán las sucesivas capas de la red.
        layers = [layer_0, layer_1, layer_2, layer_3, ..., layer_m]
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
        tf.reset_default_graph()
        
        '''self.init_dict = {'truncated_normal': tf.truncated_normal,
                          'xavier': tf.layers.xavier_initializer,
                          'he': }'''     
        self.activations_dict = {'relu': tf.nn.relu,
                                 'sigmoid': tf.nn.sigmoid,
                                 'tanh': tf.nn.tanh,
                                 'identity': tf.identity}
        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}
        self.layers = layers
        self.create_net()
    
    def create_net(self):
        def index(self, dim):
                if type(dim) is tuple:
                    index = len(dim)-1
                else:
                    index = 0
                return index
        
        def fc_layer(self, unit, dim, params):
            if index(dim) == 2: newdim = dim[0] * dim[1] * dim[2]
            elif index(dim) == 1: newdim = dim[0] * dim[1]
            else: newdim = dim
            
            activate = self.activations_dict[params['activation']]
            #init = self.init_dict[params['init']]
            init = tf.truncated_normal(weights_shape)
            with tf.name_scope('fc_layer'):
                weights_shape = (int(unit.get_shape()[1]), newdim)
                weights = tf.Variable(init, name='weights')
                bias = tf.Variable(tf.zeros(newdim), name='bias')
                activ = tf.add(tf.matmul(unit, weights), bias, name='activation')
                return activate(activ, name='unit')
        
        def conv_layer(self, unit, dim, params):
            activation = self.activations_dict[params['activation']]
            #init = self.init_dict[params['init']]
            init = tf.layers.xavier_initializer
            layer_list = [tf.layers.conv1d, tf.layers.conv2d, tf.layers.conv3d]
            return layer_list[index(dim)](inputs=unit,
                                    filters=dim,
                                    kernel_size=params['kernel_size'],
                                    strides=params['stride'],
                                    padding=params['padding'],
                                    activation=activation,
                                    kernel_initializer=init,
                                    bias_initializer=init,
                                    name='conv_layer')
        
        def maxpool_layer(self, unit, dim, params):
            layer_list = [tf.layers.max_pooling1d, tf.layers.max_pooling2d, tf.layers.max_pooling3d]
            return layer_list[index(dim)](inputs=unit,
                                    pool_size=params['kernel_size'],
                                    strides=params['stride'],
                                    padding=params['padding'],
                                    name='maxpool_layer')
                                    
        def dropout_layer(self, unit, dim, params):
            return tf.layers.dropout(inputs=unit,
                                     rate=params['prob'],
                                     name='dropout_layer')
        
        def LRN_layer(self, unit, dim, params):
            return tf.nn.local_response_normalization(
                              inputs=unit,
                              bias=params['LRN_params'][0],
                              alpha=params['LRN_params'][1],
                              beta=params['LRN_params'][2],
                              depth_radius=params['LRN_params'][3],
                              name='LRN_layer'
                              
        self.layer_dict = {'fc': fc_layer
                           'conv': conv_layer
                           'maxpool': maxpool_layer
                           'dropout': dropout_layer
                           'LRN': LRN_layer}
                                
        self.x = tf.placeholder(tf.float32, shape= self.layers[0]['dim'], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=self.layers[-1]['dim'], name='y_')
        unit = self.x;
        for type, dim, params in zip(self.layers[1: -1]['type'],
                                     self.layers[1: -1]['dim'],
                                     self.layers[1: -1]):
            layer = choose_layer(dim,type)
            unit = layer(unit, dim, params)
            
        layer = choose_layer(self.layers[-1]['dim'], self.layers[-1]['type'])
        self.y = layer(unit, self.layers[-1]['dim'], self.layers[-1]) #la ultima capa debe tener activación identity
    
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
            loss_fn = self.loss_dict[self.layers[-1]['activation']]
            self.loss = tf.reduce_mean(loss_fn(logits=self.y,
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
        '''
        El método predict recibirá datos de test y devolverá 
        la predicción de la red, una vez entrenada.
        '''
        with tf.Session() as sess:
            self.saver.restore(sess, "./MLP.ckpt")
            loss_fn = self.loss_dict[self.layers[-1]['activation']]
            pred = loss_fn(self.y)
            y_pred = sess.run(pred, feed_dict={self.x: x_test})
        return y_pred


if __name__ == "__main":
    layer_0 = {'dim': 30}
    layer_1 = {'type': 'fc', 'dim': 50, 'activation': 'sigmoid'}
    layer_2 = {'type': 'fc', 'dim': 10, 'activation': 'sigmoid'}
    layer_list = [layer_0, layer_1, layer_2]
    net = NetConstructor(layer_list);
   

 


