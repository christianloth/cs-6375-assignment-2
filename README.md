# CS6375 Assignment 2

This is for the NN assignment for CS6375 by Christian Loth.
This project predicts wine quality using a feedforward neural network.
The wine dataset contains different properties of wines, categorized as red and white.
The target variable is the quality of wine on a scale of 1 to 10.

## Requirements

- Python 3.11 or higher
- `pipenv` for managing dependencies and the virtual environment.
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Neural Network Architecture

- The neural network has one hidden layer.
- The input layer size corresponds to the number of features in the dataset.
- Activation functions available: `sigmoid`, `tanh`, and `relu`.
- Implemented momentum as the optimization technique on top of the basic backpropagation algorithm.

## Setup

1. cd into the cs-6375-assignment-2 directory:
    ```bash
    cd cs-6375-assignment-2
    ```
2. Using `pipenv` to manage dependencies and virtual environment:
    ```bash
    pipenv install
    ```

3. Activate the virtual environment:
    ```bash
    pipenv shell
    ```

## Running the Program

After setting up the environment and installing the necessary dependencies:

1. Run each portion separately inside the pipenv shell that you have activated:
    ```bash
    python assignment.py
    ```
   - Note you must wait after executing the command. It can take a bit for the plots to display depending on how fast your processor is.
   - Two plots will be displayed, a loss plot, and a weight changes plot.
   
- You can change activation functions by changing the string inside the NeuralNetwork instance. I have used all three activation functions in the program and in the plots.
- You can change the number of iterations by modifying the constant `NUM_ITERATIONS` capitalized at the very top of the program.
- We are using an 80/20 split for training and testing data.
   
## Results

The results are presented as a loss function graph for each of the activation functions on the test data.
I have also printed out the first 10 rows for predicted values vs. actual values to stdout.