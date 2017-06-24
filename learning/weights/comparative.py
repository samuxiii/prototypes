import tensorflow as tf
import matplotlib.pyplot as plt

n = 1000

A = tf.random_uniform([n])
B = tf.random_normal([n])
C = tf.truncated_normal([n])
D = tf.truncated_normal([n], stddev=0.5)

with tf.Session() as sess:
    a, b, c, d = sess.run([A, B, C, D])
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

bins_ = 100
range_ = (-5, 5)

plt.figure("Distribution Comparative")

plt.subplot(221)
plt.hist(a, bins_, range_, label="Random Uniform", color='b')
plt.legend(fontsize='small')

plt.subplot(222)
plt.hist(b, bins_, range_, label="Random Normal", color='g')
plt.legend(fontsize='small')

plt.subplot(223)
plt.hist(c, bins_, range_, label="Truncated Normal", color='r')
plt.legend(fontsize='small')

plt.subplot(224)
plt.hist(d, bins_, range_, label="Truncated Normal stddev=0.5", color='y')
plt.legend(fontsize='small')

plt.show()
