"""
    example for MNIST recognization with softmax output layer neural network. 
"""

import tensorflow as tf
import numpy as np

# load mnist data set
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.train.labels[0])

### create tensorflow structure start###
#define input and output tensor
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float",[None,10])

#define variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#define model
y = tf.nn.softmax(tf.matmul(x,W)+b)

#define cost function
cross_entropy = - tf.reduce_sum(y_*tf.log(y))

#define training parameters
learning_rate = 0.01
optimizer =  tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

#initialization operator
init = tf.global_variables_initializer()
### create tensorflow structure end###

#start session
sess = tf.Session()
sess.run(init)     #Very important!!!

#training model
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    if step%20 ==0:
        train_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        train_accuracy = tf.reduce_mean(tf.cast(train_prediction,"float"))
        train_accuracy_precent = sess.run(train_accuracy, feed_dict={x:batch_xs, y_:batch_ys})
        print("step %d training accuracy:%g"%(step,train_accuracy_precent))

#valuation
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
accuracy_precent = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
print('testing accuracy:%g'%accuracy_precent)

