import numpy as np

def input_layer(data: np.ndarray, hidden_size: int) -> np.ndarray:
    '''Takes in the data and passes an array of x(w, b) to the first layer'''

    # Find amount of features (nodes needed from data shape)
    num_features = data.shape[0]

    # Create weights and biases for data
    weight = np.random.randn(hidden_size, num_features)
    bias = np.zeros(hidden_size, 1)

    # Map data back in linear distribution
    nn_data = np.dot(weight, data) + bias

    return nn_data