import weights as w
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/tmp/datasets/tensorflow/mnist', one_hot=True)

train_features = mnist.train.images
valid_features = mnist.validation.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
valid_labels = mnist.validation.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

data_dict = { 'trf': train_features, 'vaf': valid_features, 'tef': test_features, 'trl': train_labels, 'val': valid_labels, 'tel': test_labels}

# -- #
bias = tf.Variable(tf.zeros([n_classes]))
#normal distribution
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
#bias = tf.Variable(tf.random_normal([n_classes]))
costs_1 = w.run(weights, bias, data_dict)

#zeros
weights = tf.Variable(tf.zeros([n_input, n_classes]))
#bias = tf.Variable(tf.zeros([n_classes]))
costs_2 = w.run(weights, bias, data_dict)

#ones
weights = tf.Variable(tf.ones([n_input, n_classes]))
#bias = tf.Variable(tf.ones([n_classes]))
costs_3 = w.run(weights, bias, data_dict)

#truncated normal distribution
weights = tf.Variable(tf.truncated_normal([n_input, n_classes]))
#bias = tf.Variable(tf.truncated_normal([n_classes]))
costs_4 = w.run(weights, bias, data_dict)

#plotting
plt.figure()
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.plot(costs_1, 'r-', label='normal distribution')
plt.plot(costs_2, 'y-', label='zeros')
plt.plot(costs_3, 'g-', label='ones')
plt.plot(costs_4, 'b-', label='truncated normal distribution')

plt.legend()
plt.show()