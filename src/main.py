# Import numpy (only key dependency)
import numpy as np

# Call nueral network functions
from components.build_layers import initialise_parameters
from components.foward_pass import foward_pass
from components.loss_functions import find_loss, create_loss_var

def main(data: np.ndarray = None, task_type: str = None) -> None:
    '''Run Nueral Network operations in order'''

    # Dummy data for testing 
    task_type = 'regression'
    test_layers = [10, 5, 3, 1]
    data = np.random.randn(10, 2)

    # Build dictionary of layers assigning weights and bias
    params, output_nodes = initialise_parameters(test_layers)

    # Pass the data through the layers
    y_pred = foward_pass(data, params, activation_function='relu')

    # Create variables required to calculate loss
    y_true, total_samples = create_loss_var(data, output_nodes)

    loss = find_loss(task_type, y_pred, y_true, total_samples)

    # Create dict of NN run
    run_meta = {
        'Input Shape': {data.shape},
        'Output Shape': {y_pred.shape},
        'Loss': {loss}
    }

    print(loss)

    return None

if __name__ == '__main__':
    main()