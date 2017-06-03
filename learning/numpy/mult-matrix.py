# Use the numpy library
import numpy as np


def prepare_inputs(inputs):
    input_array = np.array([inputs])
    inputs_minus_min = input_array - np.amin(input_array)
    inputs_div_max = inputs_minus_min / np.amax(inputs_minus_min)

    return input_array, inputs_minus_min, inputs_div_max
    

def multiply_inputs(m1, m2):
    #checking shapes before multiplying
    if ((m1.shape[1] != m2.shape[0]) and (m2.shape[1] != m1.shape[0])):
        return False

    if (m1.shape[1] == m2.shape[0]):
        return np.dot(m1,m2)
    else:
        return np.dot(m2,m1)
    

input_array, inputs_minus_min, inputs_div_max = prepare_inputs([-1,2,7])

print("Input: {}".format(input_array))
print("Input 'normalized': {}".format(inputs_div_max))

print("Multiply:\n{}".format(multiply_inputs(np.array([[1,2,3],[4,5,6]]), np.array([[1],[2],[3]]))))
print("Multiply:\n{}".format(multiply_inputs(np.array([[1,2,3],[4,5,6]]), np.array([[1,2]]))))

