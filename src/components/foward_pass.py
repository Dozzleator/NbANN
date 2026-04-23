import numpy as np

# Import activation functions 
from activation_functions import *

def foward_pass(data: np.ndarray, parameters: dict, activation_function:str = None) -> np.ndarray:
    '''Retrives the weights (stored in dict) and passes the data through'''

    # Find amount of layers (alls layers in dict has (w, b) so must / 2)
    num_layers = len(parameters) // 2

    activation_array = data.copy()

    for i in range(1, num_layers + 1):
        # Retrieve weights and biases for the required layer
        weight = parameters['W' + str(i)]
        bias = parameters['b' + str(i)]

        # Map redults back into array (giving linear distribution)
        mapped_data = np.dot(weight, activation_array) + bias

        # Find the step for activation activation function
        if activation_function is None:
            # By default use relu
            activation_array = ReLU(mapped_data)

    return activation_array