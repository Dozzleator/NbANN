# Import numpy (only key dependency)
import numpy as np

# Call nueral network functions
from components.build_layers import initialise_parameters
from components.foward_pass import foward_pass
from components.loss_functions import find_loss, create_loss_var
from components.back_propergation import back_propergation
from components.optimiser import optimise_parameters

def main(data: np.ndarray = None, task_type: str = None, activation_function: str = None) -> None:
    '''Run Nueral Network operations in order'''

    # Dummy data for testing 
    task_type = 'regression'
    activation_function = 'relu'
    learning_rate = 0.001
    test_layers = [10, 5, 3, 1]
    data = np.random.randn(10, 2)

    # Build dictionary of layers assigning weights and bias
    params, output_nodes = initialise_parameters(test_layers)

    # Pass the data through the layers
    y_pred, cache, num_layers = foward_pass(data, params, activation_function)

    # Create variables required to calculate loss
    y_true, total_samples = create_loss_var(data, output_nodes)

    # Find the loss activating the loss function
    loss = find_loss(task_type, y_pred, y_true, total_samples)

    # Initiate back propergation and return dict of gradiants
    gradiants = back_propergation(y_pred, y_true, cache, params, total_samples, num_layers, activation_function)

    # Call optimiser to improve model based on results
    final_params = optimise_parameters(params, gradiants, num_layers, learning_rate)

    print(final_params)

    # Create dict of NN run
    run_meta = {
        'Input Shape': {data.shape},
        'Output Shape': {y_pred.shape},
        'Loss': {loss}
    }

    return None

if __name__ == '__main__':
    main()