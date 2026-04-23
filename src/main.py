# Import numpy (only key dependency)
import numpy as np

# Call nueral network functions
from components.input_layer import initialise_parameters
from components.foward_pass import foward_pass

def main(data: np.ndarray) -> None:
    '''Run Nueral Network operations in order'''

    # Dummy data for testing 
    test_layers = [10, 5, 3, 1]
    dummy_data = np.random.randn(10, 2)

    # Build dictionary of layers assigning weights and bias
    params = initialise_parameters(test_layers)

    # Pass the data through the layers
    foward_data = foward_pass(data, params)

    return None

if __name__ == '__main__':
    main()
