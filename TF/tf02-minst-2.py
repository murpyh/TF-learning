import tensorflow as tf
import numpy as np

# load mnist data set
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#define interactive session
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#define variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#initialization
init = tf.global_variables_initializer()
sess.run(init)

#define model
y = tf.nn.softmax(tf.matmul(x,W)+b)

#define cost function
cross_entropy = - tf.reduce_sum(y_*tf.log(y))

#define training:
learning_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

#training model
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#valuation
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
accuracy_precent = accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels})
print(accuracy_precent)

