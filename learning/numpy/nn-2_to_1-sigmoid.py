#!/usr/bin/env python3.6

'''
Simple network with two input nodes and 
one output node with a sigmoid activation function.
'''
import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

# y = f(h) = sigmoid(∑(w*x + b))
output = sigmoid(inputs.dot(weights.T) + bias)

print('Output:')
print(output)
