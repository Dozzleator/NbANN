import numpy as np

def create_loss_var(data: np.ndarray, output_nodes: np.ndarray) -> np.ndarray:
    '''Create a "true" y to compare with prediction from foward prop'''

    # find amount of samples 
    num_samples = data.shape[1]

    # select from foward pass * num samples
    y_true = np.random.rand(output_nodes, num_samples)

    return y_true, num_samples

def MSE(y_pred: np.ndarray, y_true:np.ndarray, total_samples: int)-> float:
    '''Mean Squared Error (MSE) loss function to determine loss for regression'''

    # Compute squared difference then average it
    loss = (1 / (2 * total_samples)) * np.sum(np.square(y_pred - y_true))

    return loss

def Entropy(y_pred: np.ndarray, y_true:np.ndarray, total_samples: int) -> float:
    '''Binary Cross-Entropy loss function for finding loss wiin classificaition problems'''

    # Compute the binary cross-entropy loss
    loss = -(1 / total_samples) * np.sum(y_true * np.log(y_pred)) + (1 - y_true) * np.log(1 - y_pred) 

    return loss

def find_loss(task_type: str, y_pred: np.ndarray, y_true:np.ndarray, total_samples: int) -> float:
    '''Run the correct loss function for the specified task type'''

    # If regression run MSE
    if task_type == 'regression':
        loss = MSE(y_pred, y_true, total_samples)

    # If classification run Entropy
    elif task_type == 'classification':
        loss = Entropy(y_pred, y_true, total_samples)

    return loss 