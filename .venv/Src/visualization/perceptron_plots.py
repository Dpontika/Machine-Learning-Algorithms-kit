import numpy as np
import matplotlib.pyplot as plt


"""
============================================================================
                    === 2D Visualization Functions ===
    Task 1-i-a: Classify points on the plane (2D) into 2 classes - Training
    Task 1-i-b: Classify points on the plane (2D) into 2 classes - Recall
=============================================================================

"""
#Graph 1ia: Trainning patterns of the 2 classes
def plot_training_2D_data(X_train, y_train):
    """
        Plot the 2D training data points with different markers for each class.

        Parameters:
        - X_train: List or array of 2D input vectors
        - y_train: List or array of class labels (0 or 1)
    """

    for i in range(len(X_train)):
        # Get the first feature (x-axis value)
        X1 = X_train[i][0]
        # Get the second feature (y-axis value)
        X2 = X_train[i][1]

        # Get the label (either 0 or 1)
        label = y_train[i]

        # Choose color and marker depending on the label
        if label == 0:
            color = 'blue'
            marker = 'o'
            # Add legend label only once (first occurrence)
            legend_label = 'Class 0' if i == 0 else ''
        else:
            color = 'red'
            marker = 'x'
            legend_label = 'Class 1' if i == 0 else ''

        # Plot the point
        plt.scatter(X1, X2, color=color, marker=marker, label=legend_label)

    plt.title("1-i-a: 2D Training Data")
    plt.xlabel("X1 (Feature 1)")
    plt.ylabel("X2 (Feature 2)")
    plt.legend()
    plt.grid(True)
    plt.show()

#Graph 2ia: Predicted classes for each training point with decision boundary line.
def plot_decision_boundary_2D(model, X_train, y_train):
    """
        Plot the decision boundary along with classified training data.

        Parameters:
        - model: Trained Perceptron object
        - X_train: Input vectors
        - y_train: True labels
    """

    #Plot classified points
    #Blue circles for class 0, red crosses for class 1
    for i in range(len(X_train)):
        X1 = X_train[i][0]
        X2 = X_train[i][1]

        current_label = y_train[i]

        # Choose color and marker based on label
        if current_label == 0:
            point_color = 'blue'
            point_marker = 'o'
            point_label = 'Class 0' if i == 0 else ''  # Show label only once in legend
        else:
            point_color = 'red'
            point_marker = 'x'
            point_label = 'Class 1' if i == 0 else ''

        # Plot the current sample
        plt.scatter(X1, X2, color=point_color, marker=point_marker, label=point_label)

    #Plot decision boundary : w0 + w1*x1 + w2*x2 = 0 => x2 = -(w0 + w1*x1)/w2
    w = model.weights
    # To avoid division by zero, make sure the second weight (w2) is not zero
    if w[2] != 0:
        x_vals = np.linspace(0, 1, 100)
        y_vals = -(w[0] + w[1]*x_vals)/w[2]
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

        plt.title('Decision Boundary on Training Data')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True)
        plt.show()

#Graph 3ia: Output values (-1/1 or 0/1, depending on the activation function) for each training pattern.
def plot_training_outputs(model, X_train, y_train):
    """
    Plot the predicted outputs (0 or 1) for each training pattern.

    Parameters:
     - model: Trained Perceptron object
     - X_train: Input vectors
     - y_train: True label

    """
    predicted_outputs = []
    for i in range(length(X_train)):
        predicted_outputs.append(model.predict(X_train[i]))

    #Create x-axis positions for each sample (sample index)
    sample_indices = list(range(len(X_train)))

    # Plot true labels (blue circles)
    plt.scatter(sample_indices, y_train, label='True Labels', color='blue', marker='o')

    # Plot predicted labels (red crosses)
    plt.scatter(sample_indices, predicted_outputs, label='Predicted Labels', color='red', marker='x')


    plt.title("1-i-a: Perceptron Outputs vs True Labels (Training Data)")
    plt.xlabel("Sample Index")
    plt.ylabel("Output (0 or 1)")
    plt.legend()
    plt.grid(True)
    plt.show()

#Graph 1ib: Predicted vs true test labels with decision boundary
def plot_test_predictions_2D(model, X_test, y_test):
    """
    Plot predicted vs actual labels on the test set with decision boundary.

    Parameters:
    - model: Trained Perceptron object
    - X_test: List or array of 2D input vectors (test data)
    - y_test: True label
    """
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    #Make predictions for each test input using the trained model
    predictions = []
    for i in range(length(X_test)):
        predictions.append(model.predict(X_test[i]))

    for i in range(len(X_test)):
        # Get the first feature (x1) and second feature (x2)
        X1 = X_test[i][0]
        X2 = X_test[i][1]

        # Get the label (0 or 1) for this sample
        current_label = y_test[i]

        #Blue circle: true label
        if current_label == 0:
            point_label = 'True Label 0' if i == 0 else ''
        else:
            point_label = 'True Label 1' if i == 0 else ''

        plt.scatter(X1, X2, color='blue', marker='o', label=point_label)

        # Red cross: predicted label
        if predictions[i] == 0:
            point_label = 'Predicted 0' if i == 0 else ''
        else:
            point_label = 'Predicted 1' if i == 0 else ''

        plt.scatter(X1, X2, color='red', marker='x', label=point_label)


        # Draw decision boundary
        w = model.weights
        if w[2] != 0:
            x_vals = np.linspace(0, 1, 100)
            y_vals = -(w[0] + w[1] * x_vals) / w[2]
            plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

        plt.title("1-i-b: Test Predictions vs True Labels (2D)")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.grid(True)
        plt.show()

#Error curve across training epochs
def plot_errors_per_epoch(errors):
    """
        Plot the number of classification errors the Perceptron made in each training epoch.
        This helps visualize how the model improves (or struggles) during training.

        Parameters:
        - errors: A list of integers. Each value represents the number of errors the model
        made in one epoch of training.
    """

    #List of epoch numbers for the x-axis
    epoch_num = list(range(1, len(errors) + 1))

    plt.plot(epoch_num, errors, marker='o', color='purple')

    plt.title("Training Progress: Errors per Epoch")
    plt.xlabel("Epoch Number")
    plt.ylabel("Number of Errors")
    plt.grid(True)
    plt.show()


""" 
============================================================================
                    === 3D Visualization Functions ===
    Task 1-ii-a: Classification of points in 3D space into 2 classes - Training
    Task 1-ii-b: Classification of points in 3D space into 2 classes - Recall
=============================================================================

"""
#Graph 1iia: Trainning patterns of the 2 classes (3D)
def plot_training_data_3D(X_train, y_train):
    """
    Plot the 3D training data points with different markers for each class.
    Points from Class 0 are shown as blue circles, and points from Class 1 are red crosses.

    Parameters:
    - X_train: List or array of 3D input vectors (shape: [n_samples, 3])
    - y_train: List or array of class labels (0 or 1)

    """
    #Set up 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range (len(X_train)):
        X1 = X_train[i][0]
        X2 = X_train[i][1]
        X3 = X_train[i][2]

        label = y_train[i]

        if label == 0:
            point_color = 'blue'
            point_marker = 'o'
            point_label = 'Class 0' if i == 0 else ''
        else:
            point_color = 'red'
            point_marker = 'x'
            point_label = 'Class 1' if i == 0 else ''

        #Plot point in 3D
        ax.scatter(X1, X2, X3, color=point_color, marker=point_marker, label=label)

    ax.set_title("1-ii-a: 3D Training Data")
    ax.set_xlabel("X1 (Feature 1)")
    ax.set_ylabel("X2 (Feature 2)")
    ax.set_zlabel("X3 (Feature 3)")
    ax.legend()
    plt.grid(True)
    plt.show()

# Graph 2iia: Predicted classes for each training point with decision boundary line (3D)
def plot_decision_boundary_3D(model, X_train, y_train):
    """
        Plot 3D training data and the decision boundary.

        Parameters:
        - model: Trained Perceptron object with weights [bias, w1, w2, w3]
        - X_train: List or array of 3D input vectors (shape: [n_samples, 3])
        - y_train: List or array of true class labels (0 or 1)
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range (len(X_train)):
        X1 = X_train[i][0]
        X2 = X_train[i][1]
        X3 = X_train[i][2]

        predicted_label = model.predict(X_train[i])

        if predicted_label == 0:
            point_color = 'blue'
            point_marker = 'o'
            point_label = 'Predicted Class 0' if i == 0 else ''
        else:
            point_color = 'red'
            point_marker = 'x'
            point_label = 'Predicted Class 1' if i == 0 else ''

        ax.scatter(X1, X2, X3, color=point_color, marker=point_marker, label=point_label)

    w=model.weights

    if w[3] != 0:
        # Create a grid of x1 and x2 values to compute the plane
        x1_vals = np.linspace(0, 1, 10)
        x2_vals = np.linspace(0, 1, 10)
        x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

        # Compute x3 (z) values using the plane equation:
        # w0 + w1*x1 + w2*x2 + w3*x3 = 0 → solve for x3:
        # x3 = -(w0 + w1*x1 + w2*x2) / w3
        x3_grid = -(w[0] + w[1] * x1_grid + w[2] * x2_grid) / w[3]

        # Plot the plane
        ax.plot_surface(x1_grid, x2_grid, x3_grid, alpha=0.3, color='gray')

    ax.set_title("1-ii-a: 3D Decision Boundary and Predictions")
    ax.set_xlabel("X1 (Feature 1)")
    ax.set_ylabel("X2 (Feature 2)")
    ax.set_zlabel("X3 (Feature 3)")
    ax.legend()
    plt.grid(True)
    plt.show()

#Graph 1iib: Predicted vs true test labels with decision plane (3D)
def plot_test_predictions_3D(model, X_test, y_test):
    """'
        Plot predicted vs actual labels on the test set with decision boundary.

        Parameters:
        - model: Trained Perceptron object
        - X_test: List or array of 3D test input vectors
        - y_test: List or array of true labels (0 or 1)
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range (len(X_test)):
        X1 = X_test[i][0]
        X2 = X_test[i][1]
        X3 = X_test[i][2]

        true_label = y_test[i]
        predicted_label = model.predict(X_test[i])

        #Plot true label
        if true_label == 0:
            point_label = 'True Label 0' if i==0 else ''
        else:
            point_label = 'True Label 1' if i == 0 else ''
        ax.scatter(X1, X2, X3, color=blue, marker='o', label=true_label)

        # Plot the predicted label (red cross)
        if predicted_label == 0:
            point_label = 'Predicted 0' if i == 0 else ''
        else:
            point_label = 'Predicted 1' if i == 0 else ''
        ax.scatter(x1, x2, x3, color='red', marker='x', label=predicted_label_name)

    w = model.weights

    if w[3] != 0:
        # Create a grid of x1 and x2 values to compute the plane
        x1_vals = np.linspace(0, 1, 10)
        x2_vals = np.linspace(0, 1, 10)
        x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

    # Compute x3 (z) values using the plane equation:
    # w0 + w1*x1 + w2*x2 + w3*x3 = 0 → solve for x3:
    # x3 = -(w0 + w1*x1 + w2*x2) / w3
    x3_grid = -(w[0] + w[1] * x1_grid + w[2] * x2_grid) / w[3]

    # Plot the plane
    ax.plot_surface(x1_grid, x2_grid, x3_grid, alpha=0.3, color='gray')

    ax.set_title("1-ii-b: 3D Recall Phase - Predictions vs True Labels")
    ax.set_xlabel("X1 (Feature 1)")
    ax.set_ylabel("X2 (Feature 2)")
    ax.set_zlabel("X3 (Feature 3)")
    ax.legend()
    plt.grid(True)
    plt.show()





