import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

NUM_ITERATIONS = 250  # Number of iterations for training constant


class WineQualityDataPreprocessor:

    def __init__(self):
        self.red_wine = pd.read_csv('https://raw.githubusercontent.com/christianloth/cs-6375-public-files/main/wine%2Bquality/winequality-red.csv', sep=';')
        self.white_wine = pd.read_csv('https://raw.githubusercontent.com/christianloth/cs-6375-public-files/main/wine%2Bquality/winequality-white.csv', sep=';')

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        # Merge the datasets and create wine_type feature
        self.red_wine['wine_type'] = 0
        self.white_wine['wine_type'] = 1
        combined_data = pd.concat([self.red_wine, self.white_wine], axis=0)

        # Separate features and target var
        X = combined_data.drop('quality', axis=1)
        y = combined_data['quality']

        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into 80% train, 20% test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test


class NeuralNetwork:
    """
    Matrix notation:
        ss: Number of samples in the dataset.
        is: Number of features.
        hs: Number of neurons in the hidden layer.
        os: Number of neurons in the output layer.
    """

    def __init__(self, input_size, hidden_size, output_size=1, activation_function="sigmoid", gamma=0.9):

        self.Z_output = None
        self.A_hidden = None
        self.Z_hidden = None
        self.input_size = input_size  # is
        self.hidden_size = hidden_size  # hs
        self.output_size = output_size  # os

        self.activation_function = activation_function

        # Weights and biases initialization with random vals
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01  # weights from hidden to input
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01  # weights from hidden to output
        self.bias_output = np.zeros((1, self.output_size))

        self.loss_history = []  # List to store loss values over iterations

        # Here is my optimizer to satisfy project requirements: Momentum
        # Velocity terms for Momentum
        self.velocity_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.velocity_bias_hidden = np.zeros_like(self.bias_hidden)
        self.velocity_weights_hidden_output = np.zeros_like(self.weights_hidden_output)
        self.velocity_bias_output = np.zeros_like(self.bias_output)
        # Momentum gamma parameter
        self.gamma = gamma

        self.weights_input_hidden_history = []
        self.weights_hidden_output_history = []

    # Activation functions and their derivatives
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.square(self.tanh(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        dx = np.array(x, copy=True)
        dx[x <= 0] = 0  # technically DNE at x=0 though
        dx[x > 0] = 1
        return dx

    def activation(self, x):
        if self.activation_function == "sigmoid":
            return self.sigmoid(x)
        elif self.activation_function == "tanh":
            return self.tanh(x)
        elif self.activation_function == "relu":
            return self.relu(x)

    def activation_derivative(self, x):
        if self.activation_function == "sigmoid":
            return self.sigmoid_derivative(x)
        elif self.activation_function == "tanh":
            return self.tanh_derivative(x)
        elif self.activation_function == "relu":
            return self.relu_derivative(x)

    def forward_propagation(self, X):  # X dimensions: (ss, is)
        # Note: `Z` means linear output and `A` means activation here

        # Linear output for hidden layer
        # Z_hidden dimensions: (ss, hs)
        self.Z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden

        # Activation for hidden layer
        # A_hidden dimensions: (ss, hs)
        self.A_hidden = self.activation(self.Z_hidden)

        # Linear output for output layer
        # Z_output dimensions: (ss, os)
        self.Z_output = np.dot(self.A_hidden, self.weights_hidden_output) + self.bias_output

        # We don't apply activation function to the output layer because wine quality output is between 1 and 10. It's not a binary classification.
        return self.Z_output

    def backward_propagation(self, X, target_output):  # X dimensions: (ss, is), y dimensions: (ss, os)
        ss = X.shape[0]  # Number of samples

        # Compute the derivative of the loss w.r.t Net_output
        # dZ_output dimensions: (ss, os)
        dZ_output = self.Z_output - target_output  # dL/dZ

        # Compute the derivatives of loss w.r.t weights and biases between hidden and output layer
        # dW_hidden_output dimensions: (hs, os)
        # db_output dimensions: (1, os)
        dW_hidden_output = (1 / ss) * np.dot(self.A_hidden.T, dZ_output)  # dL/dW
        db_output = (1 / ss) * np.sum(dZ_output, axis=0, keepdims=True)  # dL/db

        # Compute the derivative of the loss w.r.t A_hidden
        # dA_hidden dimensions: (ss, hs)
        dA_hidden = np.dot(dZ_output, self.weights_hidden_output.T)  # dA/dZ

        # Compute the derivative of the loss w.r.t Z_hidden
        # dZ_hidden dimensions: (ss, hs)
        dZ_hidden = dA_hidden * self.activation_derivative(self.Z_hidden)  # dL/dZ

        # Compute the derivatives w.r.t weights and biases between input and hidden layer
        # dW_input_hidden dimensions: (is, hs)
        # db_hidden dimensions: (1, hs)
        dW_input_hidden = (1 / ss) * np.dot(X.T, dZ_hidden)  # dW/dZ
        db_hidden = (1 / ss) * np.sum(dZ_hidden, axis=0, keepdims=True)  # dL/db

        return dW_input_hidden, db_hidden, dW_hidden_output, db_output

    def train(self, X, target_output, epochs, learning_rate):
        for _ in range(epochs):
            # Forward propagation
            predictions = self.forward_propagation(X)

            # Compute and store the loss
            loss = self.mean_squared_error(target_output, predictions)
            self.loss_history.append(loss)

            # Backward propagation
            dW_input_hidden, db_hidden, dW_hidden_output, db_output = self.backward_propagation(X, target_output)

            # Momentum updates: weight corrections
            self.velocity_weights_input_hidden = self.gamma * self.velocity_weights_input_hidden + learning_rate * dW_input_hidden
            self.weights_input_hidden -= self.velocity_weights_input_hidden
            self.velocity_bias_hidden = self.gamma * self.velocity_bias_hidden + learning_rate * db_hidden
            self.bias_hidden -= self.velocity_bias_hidden
            self.velocity_weights_hidden_output = self.gamma * self.velocity_weights_hidden_output + learning_rate * dW_hidden_output
            self.weights_hidden_output -= self.velocity_weights_hidden_output
            self.velocity_bias_output = self.gamma * self.velocity_bias_output + learning_rate * db_output
            self.bias_output -= self.velocity_bias_output

            # Store weights for plotting
            self.weights_input_hidden_history.append(self.weights_input_hidden.copy())
            self.weights_hidden_output_history.append(self.weights_hidden_output.copy())

    def mean_squared_error(self, y_true, y_pred):
        return 1 / 2 * np.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    preprocessed_data = WineQualityDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessed_data.preprocess()
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_test_reshaped = y_test.values.reshape(-1, 1)

    learning_rates = [0.001, 0.01, 0.1]
    activation_functions = ["sigmoid", "tanh", "relu"]
    results = []

    # Figure for Loss plots
    fig_loss, axes_loss = plt.subplots(len(learning_rates), len(activation_functions), figsize=(15, 10))
    fig_loss.tight_layout(pad=5.0)
    fig_loss.subplots_adjust(top=0.9)
    fig_loss.suptitle(f'Loss with Momentum of Different Activation Functions and Learning Rates ({NUM_ITERATIONS} iterations)', fontsize=16)

    # Figure for Weight changes plots
    fig_weights, axes_weights = plt.subplots(len(learning_rates), len(activation_functions), figsize=(15, 10))
    fig_weights.tight_layout(pad=5.0)
    fig_weights.subplots_adjust(top=0.9)
    fig_weights.suptitle(f'Weight Changes with Momentum of Different Activation Functions and Learning Rates ({NUM_ITERATIONS} iterations)', fontsize=16)

    for i, lr in enumerate(learning_rates):
        for j, act_fn in enumerate(activation_functions):
            nn = NeuralNetwork(input_size=12, hidden_size=8, output_size=1, activation_function=act_fn)
            nn.train(X_train, y_train_reshaped, NUM_ITERATIONS, lr)

            predictions_train = nn.forward_propagation(X_train)
            mse_train = nn.mean_squared_error(y_train_reshaped, predictions_train)

            predictions_test = nn.forward_propagation(X_test)
            mse_test = nn.mean_squared_error(y_test_reshaped, predictions_test)

            print(f"Learning Rate: {lr}, Activation Function: {act_fn}")
            print(f"Training MSE: {mse_train:.3f}")
            print(f"Test MSE: {mse_test:.3f}")
            print("=" * 60)

            # Loss plot
            axes_loss[i, j].plot(nn.loss_history)
            axes_loss[i, j].set_title(f'Momentum Loss: LR={lr}, AF={act_fn}')
            axes_loss[i, j].set_xlabel('Iterations')
            axes_loss[i, j].set_ylabel('Loss')

            results.append((lr, act_fn, mse_train, mse_test))

            # Plot weight changes
            # for each set of weights at each iteration, this computes the sum of the absolute values of all weights in that set and plots that sum over time.
            # Plotting each individual weight with a separate line on the graph could be way too messy.
            weight_changes_input_hidden = [np.sum(np.abs(w)) for w in nn.weights_input_hidden_history]
            weight_changes_hidden_output = [np.sum(np.abs(w)) for w in nn.weights_hidden_output_history]
            axes_weights[i, j].plot(weight_changes_input_hidden, label="Input to Hidden Weights")
            axes_weights[i, j].plot(weight_changes_hidden_output, label="Hidden to Output Weights", linestyle='--')
            axes_weights[i, j].set_title(f'Weight Changes: LR={lr}, AF={act_fn}')
            axes_weights[i, j].set_xlabel('Iterations')
            axes_weights[i, j].set_ylabel('Sum of Absolute Weights')
            axes_weights[i, j].legend()

    plt.show()

    df = pd.DataFrame(results, columns=["Learning Rate", "Activation Function", "Training MSE", "Test MSE"])
    print(df)
