import numpy as np

def initialise_parameters(layers: list) -> dict:
    '''Takes in the data and passes an array of x(w, b) to the first layer'''

    # Store nodes(weights/ bias) into dict to be referenced in backproporgation
    parameters = {}

    # Create a variable to identify hoe many layers there should be
    layers_dim = len(layers)

    # Loop from 1-layer until the end of the amount of layers
    for i in range(1, layers_dim):

        # Create input aßßnd output for each node
        input_nodes = layers[i-1]
        output_nodes = layers[i]

        # To ensure randomnes among layers we enforce standard deviation to the model
        std_dev = np.sqrt(2 / input_nodes)

        # Create weights and biases for data
        weight = np.random.randn(output_nodes, input_nodes) * std_dev
    
        # Initilise with a bias array of zeros
        bias = np.zeros((output_nodes, 1))

        # Store the weights and bias into dictionary 
        parameters['W' + str(i)] = weight
        parameters['b' + str(i)] = bias

    return parameters, output_nodes
