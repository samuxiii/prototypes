import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n = 1000

A = tf.random_uniform([n])
A1 = tf.random_uniform([n], -1/np.sqrt(1000), 1/np.sqrt(1000))
B = tf.random_normal([n])
C = tf.truncated_normal([n])
D = tf.truncated_normal([n], stddev=0.5)
D1 = tf.truncated_normal([n], stddev=0.1)

with tf.Session() as sess:
    a, a1, b, c, d, d1 = sess.run([A, A1, B, C, D, D1])
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

bins_ = 100
range_ = (-5, 5)

plt.figure("Distribution Comparative")

plt.subplot(231)
plt.hist(a, bins_, range_, label="Random Uniform", color='b')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize='x-small')

plt.subplot(232)
plt.hist(a1, bins_, range_, label="Random Uniform (+/-) 1/sqrt(n)", color='c')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize='x-small')

plt.subplot(233)
plt.hist(b, bins_, range_, label="Random Normal", color='g')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize='x-small')

plt.subplot(234)
plt.hist(c, bins_, range_, label="Truncated Normal", color='r')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize='x-small')

plt.subplot(235)
plt.hist(d, bins_, range_, label="Truncated Normal stddev=0.5", color='y')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize='x-small')

plt.subplot(236)
plt.hist(d1, bins_, range_, label="Truncated Normal stddev=0.1", color='orange')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., fontsize='x-small')

plt.show()
