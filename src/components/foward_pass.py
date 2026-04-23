import numpy as np

# Import activation functions 
from components.activation_functions import *

def foward_pass(data: np.ndarray, parameters: dict, activation_function:str = None) -> tuple[np.ndarray, dict]:
    '''Retrives the weights (stored in dict) and passes the data through'''

    # Find amount of layers (alls layers in dict has (w, b) so must / 2)
    num_layers = len(parameters) // 2

    activation_array = data.copy()

    # Create cach to store mathmatical actions taken moving foward
    cache = {'A0': data}

    for i in range(1, num_layers + 1):
        # Retrieve weights and biases for the required layer
        weight = parameters['W' + str(i)]
        bias = parameters['b' + str(i)]

        # Map redults back into array (giving linear distribution)
        mapped_data = np.dot(weight, activation_array) + bias

        # Save linear data
        cache['Z' + str(i)] = mapped_data

        # Only apply to the hidden layers
        if i < num_layers:

            # Find the step for activation activation function
            if activation_function == "relu":
                activation_array = ReLU(mapped_data)
            elif activation_function == 'sigmoid':
                activation_array = sigmoid(mapped_data)
            elif activation_function == 'gelu':
                activation_array = GELU(mapped_data)

        # Keep output layer linear
        else:
            activation_array = mapped_data

        # Save output layer to cache
        cache['A' + str(i)] = activation_array

    return activation_array, cache