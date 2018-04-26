import tensorflow as tf
import numpy as np

print(tf.__version__)

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型 
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()  #Session要大写
sess.run(init)          #初始化很重要

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b))

# 结束session
sess.close()


# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]


###method2 for session operation
##with tf.Session() as sess:    # 不需要close处理
##     sess.run()


