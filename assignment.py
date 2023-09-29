import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

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
        ss: Number of samples in the dataset.
        is: Number of features (in this case, 12).
        hs: Number of neurons in the hidden layer.
        os: Number of neurons in the output layer (1 for our regression approach).
    """

    def __init__(self, input_size, hidden_size, output_size, activation_function="sigmoid"):

        # Network architecture
        self.Z_output = None
        self.A_hidden = None
        self.Z_hidden = None
        self.input_size = input_size  # is
        self.hidden_size = hidden_size  # hs
        self.output_size = output_size  # os

        # Activation function selection
        self.activation_function = activation_function

        # Weights and biases initialization
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01  # weights from hidden to input
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01  # weights from hidden to output
        self.bias_output = np.zeros((1, self.output_size))

        self.loss_history = []  # List to store loss values over epochs

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
        dx[x <= 0] = 0
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

    def forward_propagation(self, X):  # X: (ss, is)
        # Linear output for hidden layer
        # Z_hidden: (ss, hs)
        self.Z_hidden = np.dot(X, self.weights_input_hidden) + self.bias_hidden

        # Activation for hidden layer
        # A_hidden: (ss, hs)
        self.A_hidden = self.activation(self.Z_hidden)

        # Linear output for output layer
        # Z_output: (ss, os)
        self.Z_output = np.dot(self.A_hidden, self.weights_hidden_output) + self.bias_output

        # We don't apply activation function to the output layer because wine quality output is between 1 and 10. It's not a binary classification.
        return self.Z_output

    def backward_propagation(self, X, target_output):  # X: (ss, is), y: (ss, os)
        ss = X.shape[0]  # Number of samples

        # Compute the derivative of the loss w.r.t Net_output
        # dZ_output: (ss, os)
        dZ_output = self.Z_output - target_output

        # Compute the derivatives of loss w.r.t weights and biases between hidden and output layer
        # dW_hidden_output: (hs, os)
        # db_output: (1, os)
        dW_hidden_output = (1 / ss) * np.dot(self.A_hidden.T, dZ_output)  # dL/dW
        db_output = (1 / ss) * np.sum(dZ_output, axis=0, keepdims=True)  # dL/db

        # Compute the derivative of the loss w.r.t A_hidden
        # dA_hidden: (ss, hs)
        dA_hidden = np.dot(dZ_output, self.weights_hidden_output.T)

        # Compute the derivative of the loss w.r.t Z_hidden
        # dZ_hidden: (ss, hs)
        dZ_hidden = dA_hidden * self.activation_derivative(self.Z_hidden)

        # Compute the derivatives w.r.t weights and biases between input and hidden layer
        # dW_input_hidden: (is, hs)
        # db_hidden: (1, hs)
        dW_input_hidden = (1 / ss) * np.dot(X.T, dZ_hidden)
        db_hidden = (1 / ss) * np.sum(dZ_hidden, axis=0, keepdims=True)

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

            # Update weights and biases
            self.weights_input_hidden -= learning_rate * dW_input_hidden
            self.bias_hidden -= learning_rate * db_hidden
            self.weights_hidden_output -= learning_rate * dW_hidden_output
            self.bias_output -= learning_rate * db_output

    def mean_squared_error(self, y_true, y_pred):
        """Computes the Mean Squared Error AKA Loss."""
        return 1/2 * np.mean((y_true - y_pred) ** 2)

    def plot_loss(self):
        """Plot the loss over epochs."""
        plt.plot(self.loss_history)
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def predict(self, X):
        """Predicts the output using forward propagation."""
        return self.forward_propagation(X)

# Test the initialization
nn = NeuralNetwork(input_size=12, hidden_size=8, output_size=1)


if __name__ == "__main__":
    preprocessed_data = WineQualityDataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessed_data.preprocess()
    nn = NeuralNetwork(input_size=12, hidden_size=15, output_size=1, activation_function="sigmoid")

    # 3. Train the neural network
    epochs = 1000
    learning_rate = 0.01
    nn.train(X_train, y_train.values.reshape(-1, 1), epochs, learning_rate)

    # 4. Predict on test data
    predictions = nn.forward_propagation(X_test)
    
    #print(f"Predictions: {predictions}")
    #print(f"Actual: {y_test.values.reshape(-1, 1)}")
    # do this but in a for loop printing predictions and actual next to eachother
    actuals = y_test.values.reshape(-1, 1)

    for i in range(len(predictions)):
        print("(Prediction, Actual): ({}, {})".format(predictions[i][0], actuals[i][0]))

    # 5. Evaluate the performance
    mse = nn.mean_squared_error(y_test.values.reshape(-1, 1), predictions)
    print(f"Mean Squared Error on Test Data: {mse}")

    # Plot the loss
    nn.plot_loss()