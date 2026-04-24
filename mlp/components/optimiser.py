def optimise_parameters(parameters: dict, gradiants: dict, num_layers: int, learning_rate: float = None) -> dict:
    '''Update the weights and biases in the layers using gradiant descent'''

    # Loop through each layer and find optimal weights / bias
    for i in range(1, num_layers + 1):

        # Update weights: W = W - (learning rate * error)
        parameters['W' + str(i)] = parameters['W' + str(i)] - learning_rate * gradiants['dW' + str(i)]

        # Update bias: b = b - (learning rate * error)
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * gradiants['db' + str(i)]

    return parameters