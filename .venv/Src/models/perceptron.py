"""
    This class implements a Perceptron, a simple type of artificial neuron used for binary classification tasks.
    The Perceptron learns to classify input data by adjusting its weights based on the error between
    the predicted and actual outputs

    Attributes:
   -----------
   weights : ndarray
       List of weights including the bias weight.
   activation : str
       Activation function to use ('step' or 'sign').
   input_dim : int
       The number of input dimensions (excluding the bias term).

    """

import numpy as np

class Perceptron:

    def __init__(self, input_dim, activation='step'):
        self.weights = [0.0] * (input_dim + 1)  # +1 for bias weight
        self.activation = activation
        self.input_dim = input_dim

    def activate(self, x):
        """
        Apply the activation function.

        Parameters:
        - x: A single numeric input

        Returns:
        - Output after applying the activation function (0/1 or -1/1)
        """
        if self.activation == 'step':
            return 1 if x >= 0 else 0
        elif self.activation == 'sign':
            return 1 if x >= 0 else -1

    def predict(self, x):
        """
        Predict the class label for a single input vector.

        Parameters:
        - x: Input vector (without bias term)

        Returns:
        - Predicted class label
        """
        x_with_bias = np.insert(x, 0, 1)  # Add bias input = 1
        total = 0
        for i in range(len(self.weights)):
            total += self.weights[i] * x_with_bias[i]  # Weighted sum : w0 * 1 + w1 * x1 + w2 * x2 + ... + wn * xn
        return self.activate(total)

    def train(self, X_train, y_train, epochs=100, lr=0.1):
        """
        Train the perceptron using labeled training data.

        Parameters:
        - X_train: List of input vectors
        - y_train: Corresponding list of labels
        - epochs: Number of training iterations
        - lr: Learning rate

        Returns:
        - A list with number of errors in each epoch
        """
        errors_per_epoch = []

        for epoch in range(epochs):
            errors = 0
            for index in range(len(X_train)):
                x = X_train[index]
                target = y_train[index]

                prediction = self.predict(x)
                error = target - prediction
                update = lr * error

                if update != 0:
                    self.weights[0] += update  # Update bias
                    for i in range(len(x)):
                        self.weights[i + 1] += update * x[i]  # Update other weights
                    errors += 1  # Count misclassifications

            errors_per_epoch.append(errors)  # Store errors this epoch

        return errors_per_epoch

    def recall(self, X_test, y_test):
        """
        Test the perceptron on new data and calculate accuracy.

        Parameters:
        - X_test: List of input vectors
        - y_test: Corresponding true labels

        Returns:
        - predictions: List of predicted labels
        - accuracy: Proportion of correct predictions
        """
        predictions = []
        correct_count = 0

        for i in range(len(X_test)):
            x = X_test[i]
            target = y_test[i]

            prediction = self.predict(x)
            predictions.append(prediction)

            if prediction == target:
                correct_count += 1

        accuracy = correct_count / len(y_test)
        return predictions, accuracy

# # Example training data (3D)
# X_train = np.array([[2, 2, 3], [2, 3, 4], [3, 2, 2], [4, 2, 1]])
# y_train = np.array([0, 1, 0, 1])
#
# # Example test data (3D)
# X_test = np.array([[1, 2, 3], [2, 3, 4], [3, 1, 2], [2, 2, 1]])
# y_test = np.array([0, 1, 0, 1])
#
# # Initialize the Perceptron
# perceptron = Perceptron(input_dim=3, activation='sign')
#
# # Train the Perceptron
# perceptron.train(X_train, y_train, epochs=2, lr=0.1)
#
# # Evaluate the Perceptron
# predictions, accuracy = perceptron.recall(X_test, y_test)
# print("Predictions:", predictions)
# print("Accuracy:", accuracy)