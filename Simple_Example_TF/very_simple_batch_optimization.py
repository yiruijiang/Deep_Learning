import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1,1]))

my_output = tf.matmul(x_data, A)
loss = tf.reduce_mean(tf.square(my_output - y_target))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.35)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

batch_size = 15

losses = []

for i in range(10):
    k = np.random.choice(100, size = batch_size)

    x_i = np.transpose([x_vals[k]])
    y_i = np.transpose([y_vals[k]])

    sess.run(train_step, feed_dict = {x_data: x_i, y_target: y_i})
    l = sess.run(loss, feed_dict = {x_data: x_i, y_target: y_i})
    losses.append(l)
    print("Loss of %s epoch is "%i, l)
    print(sess.run(A))

plt.plot(np.arange(0,10), losses)



