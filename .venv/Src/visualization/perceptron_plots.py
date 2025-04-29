import numpy as np
import matplotlib.pyplot as plt

"""

1-i-a: Classify points on the plane (2D) into 2 classes - Training
    Graph 1a: Trainning patterns of the 2 classes
    Graph 2a: Predicted classes for each training point with decision boundary line (w0 + w1*x1 + w2*x2 = 0).
    Graph 3a: Output values (-1/1 or 0/1, depending on the activation function) for each training pattern.

1-i-b: Classify points on the plane (2D) into 2 classes - Recall
    Graph 1b: Trainning patterns of the 2 classes

"""
#Graph 1a
def plot_training_2D_data(X_train, y_train):
    """
       Plot the 2D training data with different colors/symbols for each class.

       Parameters:
       - X_train: 2D list or array of input vectors
       - y_train: List or array of class labels (0 or 1)
    """
    # Convert to numpy arrays for easier indexing
    X_train = np.array(X_train)
    y_train - np.array(y_train)

    #Class 0 - blue circles
    plt.scatter(X_train [y_train==0][:,0], X_train[y_train==0][:,1], color='blue', label='Class 0', marker='o')

    #Class 1 - red crosses
    plt.scatter(X_train [y_train==1][:,0], X_train[y_train==1][:,1], color='red', label='Class 1', marker='x')

    plt.title('Training Data (2D)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    plt.show()

#Graph 2a
def plot_decision_boundary_2D(model, X_train, y_train):
    """
          Plot the decision boundary of a trained Perceptron along with classified training data.

          Parameters:
            - model: Trained Perceptron object
            - X_train: Input vectors
            - y_train: True labels

       """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    #Plot classified points
    for i in range(length(X_train)):
        pred = model.predict(X_train[i])
        color='blue' if pred==0 else 'red'
        marker='o' if pred==0 else 'x'
        plt.scatter(X_train[i][:,0], X_train[i][:,1], color=color, marker=marker)

    #Plot decision boundary : w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1)/w2
    w = model.weights
    if w[2] != 0:
        x_vals = np.linespace(0, 1, 100)
        y_vals = -(w[0] + w[1]*x_vals)/w[2]
        plt.plot(x_vals, y_vals, color='k--', label='Decision Boundary')

        plt.title('Decision Boundary on Training Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True)
        plt.show()

#Graph 3a
def plot_training_outputs(model, X_train, y_train):
    """
     Plot the predicted outputs (0 or 1) for each training pattern.

     Parameters:
     - model: Trained Perceptron object
     - X_train: Input vectors
     - y_train: True label

     """
    outputs = []
    for i in range(length(X_train)):
        outputs.append(model.predict(X_train[i]))

    #Visualization of the output for each value
    plt.scatter(range(len(outputs)), outputs, label='Predicted', color='red', marker='x')
    plt.scatter(range(len(y_train)),y_train,color='blue', label='True', marker='o')

    plt.title('Perceptron Outputs vs True Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('Output (0 or 1)')
    plt.legend()
    plt.grid(True)
    plt.show()

#Graph 1b
def plot_test_predictions_2D(model, X_test, y_test):
        """
        Plot predicted vs actual labels on the test set with decision boundary.

        Parameters:
        - model: Trained Perceptron object
        - X_test: Test input vectors
        - y_test: Test labels
        """
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        predictions = [model.predict(x) for x in X_test]

        # Blue = True labels, Red = Predicted
        plt.scatter(X_test[:, 0], X_test[:, 1], c='blue', label='True Labels', marker='o')
        plt.scatter(X_test[:, 0], X_test[:, 1], c='red', label='Predictions', marker='x')

        # Draw decision boundary
        w = model.weights
        if w[2] != 0:
            x_vals = np.linspace(0, 1, 100)
            y_vals = -(w[0] + w[1] * x_vals) / w[2]
            plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

        plt.title("1-i-b: Recall Phase - Predictions vs True Labels")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.grid(True)
        plt.show()

