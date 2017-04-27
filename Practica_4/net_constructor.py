def NetConstructor(object):
    def __init__(self, layers):
        self.activations_dict = {'relu': tf.nn.relu,
                                 'sigmoid': tf.nn.sigmoid,
                                 'tanh': tf.nn.tanh,
                                 'identity': tf.identity}
        self.loss_dict = {'softmax': tf.nn.softmax_cross_entropy_with_logits,
                          'identity': tf.nn.l2_loss,
                          'sigmoid': tf.nn.sigmoid_cross_entropy_with_logits}
        self.layer_dict = {'fc': tf.contrib.layers.fully_connected,
                           'conv': tf.contrib.layers.conv2d,
                           'maxpool': tf.contrib.layers.max_pool2d,
                           'dropout': tf.contrib.layers.dropout,
                           'LRN': }
        self.
        self.create_net()

    def create_net(self):
        pass
    
    def train(self, x_train, t_train,
              nb_epochs=1000,
              batch_size=10,
              method=('adam', method_params)
              seed=seed_nb):
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
layer_list es una lista de diccionarios que describirán las sucesivas capas de la red.

layer_list = [layer_0, layer_2, layer_3, ..., layer_m]

layer_0 contendrá solamente la dimensión de los datos de entrada, que será una tupla o un número:

layer_0 = {'dim': (dim_0, dim_1, ..., dim_L)}

Por ejemplo,

layer_0 = {'dim': (224, 224, 3)}

en el caso de que los datos de entrada sean imágenes de dimensión 224 x 224 y 3 canales de color o

layer_0 = {'dim': 784} en el caso de que sean vectores que representen imágenes de MNIST.

En las restantes capas, la estructura será la siguiente:

(se indican los parámetros mínimos que deberán estar implementados, se
 pueden añadir otros si se desea. No todos los parámetros deben
 aparecer siempre, por ejemplo, una capa de dropout sólo necesita la
 probabilidad de hacer dropout)

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

El método predict recibirá datos de test y devolverá la predicción de la red, una vez entrenada.
'''
