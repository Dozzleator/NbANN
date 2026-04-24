import numpy as np

# Import derivitives of activation functions
from components.activation_functions import ReLU_derivative, sigmond_derivative, GELU_derivative

def back_propergation(
        y_pred: np.ndarray, 
        y_true: np.ndarray, 
        cache: dict, 
        parameters: dict, 
        total_samples: int,
        num_layers: int,
        activation_function: str = None
        ) -> np.ndarray:
    '''Calcuate the gradients and biases for problem (working backwards)'''

    # Track the calculated gradient for predictions
    gradient = {}

    # Calculate the final error of the output layer
    final_error = y_pred - y_true

    # Create loop to search backwards through the layers (Output -> Hidden -> Input)
    for i in reversed(range(1, num_layers + 1)):

        # Get the output from previous layer
        prev_out = cache['A' + str(i - 1)]

        # Calculate the grrafiant for the current layers weights and biases
        gradient['dW' + str(i)] = (1 / total_samples) * np.dot(final_error, prev_out.T)
        gradient['db' + str(i)] = (1 / total_samples) * np.sum(final_error, axis=1, keepdims=True)

        # Calculate the error to pass backwards into the next hidden layer
        if i > 1:
            current_weight = parameters['W' + str(i)]
            prev_error = cache['Z' + str(i - 1)]

            # Find the derivative of the loss function used in the hidden layer
            if activation_function == 'relu':
                derivative = ReLU_derivative(prev_error)
            elif activation_function == 'sigmoid':
                derivative = sigmond_derivative(prev_error)
            elif activation_function == 'gelu':
                derivative = GELU_derivative(prev_error)

            # Find final error of previous layer using chain logic
            final_error = np.dot(current_weight.T, final_error) * derivative

    return gradient
