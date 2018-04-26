import tensorflow as tf
import numpy as np


'''
    define a function to add layer to network
'''
def add_layer(inputs, in_size, out_size, activation_function=None):
    '''define W and b '''
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) +0.1)
    '''feedward calulation'''
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    '''activation the result'''
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


'''
    main
'''
print('Neural Network with TensorFlow!!')
