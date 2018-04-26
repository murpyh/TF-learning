"""
      MNIST recognization with convolution neural network.
"""

import tensorflow as tf
import time

# load mnist data set
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


'''
    network model parameters initialization
'''
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)  #从截断的正态分布中取出随机值
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

'''
    convolution and max pooling
'''
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

#define interactive session
sess = tf.InteractiveSession()

#define placeholder for input images and output labels
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

###construct Convolution Neural Network###
x_image = tf.reshape(x, [-1,28,28,1])

# layer1:
'''
    input: 28x28  image
    convolutional kernel:32  5*5filters, stride =1, padding=0, 2x2 max-pooling
    output: 14x14 image 
'''
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#layer2:
'''
    input: 14x14x32  image
    convolutional kernel:64  5*5filters, stride =1, padding=0, 2x2 max-pooling
    output: 7x7x64 image 
'''
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2= max_pool_2x2(h_conv2)

# fully-connected layer
'''
    input: 7x7x64  image
    layer neurons: 1024
    output: 1024 neurons 
'''
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#define cost function
cross_entropy = - tf.reduce_sum(y_*tf.log(y_conv))

#define training
'''
    Adam: 99.2%
    GradientDescent: 98.8%
'''
learning_rate = 0.0001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#valuation
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

#initialization
init = tf.global_variables_initializer()
sess.run(init)

#training model
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# testdata for validation
for i in range(10):
    test_batch = mnist.test.next_batch(1000)
    test_accuracy=accuracy.eval(feed_dict={ x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
    # time format: yy-mm-dd hh:mm:ss
    t= time.localtime()
    current_time =  time.strftime("%Y-%m-%d %H:%M:%S:", t)
    print (current_time+"test accuracy %g" %test_accuracy) 

# close session after testing 
sess.close()



