import tensorflow as tf
import numpy as np
import math

def batches(batch_size, features, labels):
    assert len(features) == len(labels)
    outout_batches = []
    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)
        
    return outout_batches


def run(weights, bias, data_dict):
    n_input = 784  # MNIST data input (img shape: 28*28)
    n_classes = 10  # MNIST total classes (0-9 digits)

    train_features = data_dict['trf']
    valid_features = data_dict['vaf']
    test_features = data_dict['tef']

    train_labels = data_dict['trl']
    valid_labels = data_dict['val']
    test_labels = data_dict['tel']

    # Features and Labels
    features = tf.placeholder(tf.float32, [None, n_input])
    labels = tf.placeholder(tf.float32, [None, n_classes])

    # Model => Logits = xW + b
    logits = tf.add(tf.matmul(features, weights), bias)

    # Define loss and optimizer
    learning_rate = tf.placeholder(tf.float32)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    batch_size = 128
    epochs = 10
    learn_rate = 0.001

    train_batches = batches(batch_size, train_features, train_labels)

    #run
    costs = []

    with tf.Session() as sess:
        sess.run(init)
        
        # Training
        for epoch_i in range(epochs):
            for batch_features, batch_labels in train_batches:
                train_feed_dict = {
                    features: batch_features,
                    labels: batch_labels,
                    learning_rate: learn_rate}
                sess.run(optimizer, feed_dict=train_feed_dict)

            # Store the cost an epoch
            current_cost = sess.run(
                cost,
                feed_dict={features: batch_features, labels: batch_labels})
            
            costs.append(current_cost)

    return costs
