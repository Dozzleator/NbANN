import numpy as np

def ReLU(data: np.ndarray) -> np.ndarray:
    '''Activation function for ReLU strategy
    - Finding the max of n
    '''

    # Activaton function for Relu (y = max(0, n))
    activatation_data = np.maximum(0, data)

    return activatation_data

def ReLU_derivative(data: np.ndarray) -> np.ndarray:
    '''Derivitive of ReLU transformation'''
    return np.where(data > 0, 1, 0)

def sigmoid(data: np.ndarray) -> np.ndarray:
    '''Activation function for Sigmoid strategy
    - Squishing everything between 1 and 0
    '''

    # Activation function for sigmoid (y = 1 / (1 + exp(-x)))
    activation_data = 1 / (1 + np.exp(-data))

    return activation_data

def sigmond_derivative(data: np.ndarray) -> np.ndarray:
    '''Derivitive of ReLU transformation'''
    sig = sigmoid(data)
    return sig * (1 - sig)

def GELU(data: np.ndarray) -> np.ndarray:
    '''Activation function for GELU
    - n forms gaussian distribution
    '''

    # Set constrants for mathmatical theorim
    pi_sqr = np.sqrt(2 / np.pi)
    coeff = 0.044715

    # Activation function for GELU
    inner = pi_sqr * (data + coeff * np.power(data, 3))
    activation_data = 0.5 * data * (1 + np.tanh(inner))

    return activation_data

def GELU_derivative(data: np.ndarray) -> np.ndarray:
    '''Derivitive of GELU transformation'''

    # Set constrants for mathmatical theorim
    pi_sqr = np.sqrt(2 / np.pi)
    coeff = 0.044715

    # Calculate inner polynomial
    inner = pi_sqr * (data + coeff * np.power(data, 3))
    tahn_inner = np.tanh(inner)

    # Calculate the derivative of the inner polynomial
    inner_derivate = pi_sqr * (1 + 3 * coeff * np.square(data))

    # Combine using product and chain theorim
    derivite = 0.5 * (1 + tahn_inner) + 0.5 * data * (1 - np.square(tahn_inner)) * inner_derivate

    return derivite