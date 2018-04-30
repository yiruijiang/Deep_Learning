import numpy as np
import tensorflow as tf
import matplotlib as plt

x_c1 = np.random.normal(-1, 0.2,200)
x_c2 = np.random.normal(3, 0.2, 200)

y_c1 = np.zeros(shape = [200,])
y_c2 = np.ones(shape = [200,])

#x_c = np.hstack([x_c1, x_c2])
x_c = np.concatenate([x_c1, x_c2])
y_c = np.squeeze(np.concatenate((y_c1, y_c2)))

A = tf.Variable(tf.random_normal(mean = 10,shape = [1]))

x = tf.placeholder(dtype = tf.float32, shape = [1])
y = tf.placeholder(dtype = tf.float32, shape = [1])

output = 1 / (1 + tf.exp(-(x + A)))

loss = - (tf.multiply(y, tf.log(output + 1e-5)) + tf.multiply(1 - y, tf.log(1 - output + 1e-5)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.25)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

losses = []
indices = []
a_s = []

for i in range (800):
    k = np.random.choice(400)
    x_i = [x_c[k]]
    y_i = [y_c[k]]

    sess.run(train, feed_dict = {x: x_i, y: y_i})

    if i % 50 == 0:
        l = np.squeeze(sess.run(loss, feed_dict= {x: x_i, y: y_i}))
        a = sess.run(A)
        print("loss of %s echo is: "%i, l)
        print("A of %s echo is "%i, a)
        indices.append(i)
        losses.append(l)
        a_s.append(a)

plt.pyplot(indices, losses)