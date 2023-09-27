import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class WineQualityDataPreprocessor:

    def __init__(self):
        # Load the datasets
        self.red_wine = pd.read_csv('https://raw.githubusercontent.com/christianloth/cs-6375-public-files/main/wine%2Bquality/winequality-red.csv', sep=';')
        self.white_wine = pd.read_csv('https://raw.githubusercontent.com/christianloth/cs-6375-public-files/main/wine%2Bquality/winequality-white.csv', sep=';')

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess(self):
        # Combine the datasets and create wine_type feature
        self.red_wine['wine_type'] = 0
        self.white_wine['wine_type'] = 1
        combined_data = pd.concat([self.red_wine, self.white_wine], axis=0)

        # Separate features and target variable
        X = combined_data.drop('quality', axis=1)
        y = combined_data['quality']

        # Normalize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split data into training and testing sets (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        return self.X_train, self.X_test, self.y_train, self.y_test


class NeuralNetwork:
    """
    A basic feedforward neural network class.

    Attributes:
        input_size (int): Number of neurons in the input layer.
        hidden_size (int): Number of neurons in the hidden layer.
        output_size (int): Number of neurons in the output layer.
        activation_function (str): Type of activation function to be used in the hidden layer.

    Matrix notation:
        m: Number of samples in the dataset.
        n: Number of features (in this case, 12).
        h: Number of neurons in the hidden layer.
        o: Number of neurons in the output layer (1 for our regression approach).
    """

    def __init__(self, input_size, hidden_size, output_size, activation_function="sigmoid"):

        # Network architecture
        self.input_size = input_size  # n
        self.hidden_size = hidden_size  # h
        self.output_size = output_size  # o

        # Activation function selection
        self.activation_function = activation_function

        # Weights and biases initialization
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01  # weights from hidden to input
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01  # weights from hidden to output
        self.bias_output = np.zeros((1, self.output_size))

    # Activation functions and their derivatives
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.square(self.tanh(z))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        dz = np.array(z, copy=True)
        dz[z <= 0] = 0
        dz[z > 0] = 1
        return dz

    def activation_derivative(self, z):
        if self.activation_function == "sigmoid":
            return self.sigmoid_derivative(z)
        elif self.activation_function == "tanh":
            return self.tanh_derivative(z)
        elif self.activation_function == "relu":
            return self.relu_derivative(z)

    def forward_propagation(self, X):  # X: (m, n)
        # Linear output for hidden layer
        # Z_hidden: (m, h)
        self.Z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden

        # Activation for hidden layer
        # A_hidden: (m, h)
        self.A_hidden = self.activation(self.Z_hidden)

        # Linear output for output layer
        # Z_output: (m, o)
        self.Z_output = np.dot(self.A_hidden, self.weights_hidden_output) + self.bias_output

        return self.Z_output

    def backward_propagation(self, X, y):  # X: (m, n), y: (m, o)
        m = X.shape[0]  # Number of samples

        # Compute the derivative of the loss w.r.t Z_output
        # dZ_output: (m, o)
        dZ_output = self.Z_output - y

        # Compute the derivatives w.r.t weights and biases between hidden and output layer
        # dW_hidden_output: (h, o)
        # db_output: (1, o)
        dW_hidden_output = (1 / m) * np.dot(self.A_hidden.T, dZ_output)
        db_output = (1 / m) * np.sum(dZ_output, axis=0, keepdims=True)

        # Compute the derivative of the loss w.r.t A_hidden
        # dA_hidden: (m, h)
        dA_hidden = np.dot(dZ_output, self.weights_hidden_output.T)

        # Compute the derivative of the loss w.r.t Z_hidden
        # dZ_hidden: (m, h)
        dZ_hidden = dA_hidden * self.activation_derivative(self.Z_hidden)

        # Compute the derivatives w.r.t weights and biases between input and hidden layer
        # dW_input_hidden: (n, h)
        # db_hidden: (1, h)
        dW_input_hidden = (1 / m) * np.dot(X.T, dZ_hidden)
        db_hidden = (1 / m) * np.sum(dZ_hidden, axis=0, keepdims=True)

        return dW_input_hidden, db_hidden, dW_hidden_output, db_output

    def train(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            # Forward propagation
            self.forward_propagation(X)

            # Backward propagation
            dW_input_hidden, db_hidden, dW_hidden_output, db_output = self.backward_propagation(X, y)

            # Update weights and biases
            self.weights_input_hidden -= learning_rate * dW_input_hidden
            self.bias_hidden -= learning_rate * db_hidden
            self.weights_hidden_output -= learning_rate * dW_hidden_output
            self.bias_output -= learning_rate * db_output


# Test the initialization
nn = NeuralNetwork(input_size=12, hidden_size=8, output_size=1)


if __name__ == "__main__":
    wine_nn = WineQualityDataPreprocessor()
    X_train, X_test, y_train, y_test = wine_nn.preprocess()
    nn = NeuralNetwork(input_size=12, hidden_size=8, output_size=1, activation_function="sigmoid")

