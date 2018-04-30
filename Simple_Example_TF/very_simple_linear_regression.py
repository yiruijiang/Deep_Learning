import numpy as np
import tensorflow as tf

sess = tf.Session()

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10,100)

x_data = tf.placeholder(shape = [1], dtype = tf.float32)
y_target = tf.placeholder(shape = [1], dtype = tf.float32)

A = tf.Variable(tf.random_normal(shape = [1]))

output = tf.multiply(A, x_data)

loss = tf.multiply(output - y_target, output - y_target)

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess.run(init)

for _ in range(1,10):
    k = np.random.choice(100)
    sess.run(train, feed_dict={x_data: [x_vals[k]], y_target: [y_vals[k]]})
    print("Loss is %s" %str(sess.run(loss,  feed_dict={x_data: [x_vals[k]], y_target: [y_vals[k]]})))
    print(sess.run(A))

# print(y_vals)