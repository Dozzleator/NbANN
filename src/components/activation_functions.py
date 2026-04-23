import numpy as np

def ReLU(data: np.ndarray) -> np.ndarray:
    '''Activation function for ReLU strategy'''

    # Activaton function for Relu (y = max(0, n))
    activatation_data = np.maximum(0, data)

    return activatation_data