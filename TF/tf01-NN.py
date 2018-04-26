'''
     a function is defineed to add layer to the network, and the network is used to predict the data.
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

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

#define test data
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder("float",[None,1])
ys = tf.placeholder("float",[None,1])

#define network
layer_1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(layer_1, 10, 1, activation_function = None)

#define loss function
loss =tf.reduce_mean( tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

#define training parameter
eta = 0.1
train_step = tf.train.GradientDescentOptimizer(eta).minimize(loss)

#initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# figure
fig = plot.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plot.ion()    #python3  command is used to continue the program after plot.show
plot.show()

# training
for step in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    # to visualize the result and improvement
    if step%50==0:
        print('Loss:%g'%sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predictvalue = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
        #plot predictin value
        lines = ax.plot(x_data, predictvalue,'r-', lw=5 )
        plot.pause(0.1)
        
sess.close()

