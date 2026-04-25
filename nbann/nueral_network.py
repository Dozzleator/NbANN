# Import numpy (only key dependency)
import numpy as np

# Call nueral network functions (Check related files to see scripts for each step)
from .components.build_layers import initialise_parameters
from .components.foward_pass import foward_pass
from .components.loss_functions import find_loss
from .components.back_propergation import back_propergation
from .components.optimiser import optimise_parameters

class NeuralNetwork:

    def __init__(self, layers: list, activation_function: str, learning_rate: float, max_iter: int, task_type: bool):
        '''Initialise the Nueral Network with specific hyperparameters'''

        # Initialise Hyperparameters
        self.layers = layers
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.epoch = max_iter
        self.task_type = task_type

        # Initilise network architecture
        self.parameters , self.output_nodes = initialise_parameters(layers)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool) -> None:
        '''Fit the model with with input data'''

        # Find total number of samples from array
        num_samples = X.shape[1]

        # Loop through defined iterations (of full runs)
        for epoch in range(self.epoch):

            # Pass the data through the layers
            y_pred, cache, num_layers = foward_pass(X, self.parameters, self.activation_function)

            # Find the loss activating the loss function
            loss = find_loss(self.task_type, y_pred, y, num_samples)

            # Initiate back propergation and return dict of gradiants
            gradiants = back_propergation(y_pred, y, cache, self.parameters, num_samples, num_layers, self.activation_function)

            # Call optimiser to improve model based on results
            self.parameters = optimise_parameters(self.parameters, gradiants, num_layers, self.learning_rate)

            # Show nueral network progress if verbose is true
            if verbose and epoch % 100 == 0:
                print(f'Epoch {epoch:4d} | Loss: {loss:.6f}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''Generate predictions for new unseen data'''

        # Run the foward pass only (we only need y_pred)
        predictions, _, _ = foward_pass(X, self.parameters, self.activation_function)

        return predictions

    def calculate_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        '''Calculate teh percentage of correct binary predications'''

        if self.task_type == 'classification': 

            # Convert continouse probabilities into binary predictions
            binary_predictions = np.where(y_pred > 0.5, 1.0, 0.0)

            # Compare to actual y and return mean of matches
            accuracy = np.mean(binary_predictions == y_true) * 100

            return float(accuracy)

        elif self.task_type == 'regression':

            # Calculate regression with mean absolute error
            mean_absolute_error = np.mean(np.abs(y_pred - y_true))

            return float(mean_absolute_error)