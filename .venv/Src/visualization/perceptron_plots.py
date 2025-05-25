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
    # Track if we've added each label to the legend yet
    class0_plotted = False
    class1_plotted = False

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
            legend_label = 'Class 0' if not class0_plotted else None
            class0_plotted = True
        else:
            color = 'red'
            marker = 'x'
            legend_label = 'Class 1' if not class1_plotted else None
            class1_plotted = True

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

    # Track if we've added each label to the legend yet
    class0_plotted = False
    class1_plotted = False

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
            legend_label = 'Class 0' if not class0_plotted else None
            class0_plotted = True
        else:
            color = 'red'
            marker = 'x'
            legend_label = 'Class 1' if not class1_plotted else None
            class1_plotted = True

        # Plot the point
        plt.scatter(X1, X2, color=color, marker=marker, label=legend_label)

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
    for i in range(len(X_train)):
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

    plt.figure(figsize=(8, 6))

    # Plot TRUE labels - just show first of each type in legend
    true_0_plotted = False
    true_1_plotted = False
    predicted_plotted = False

    for i in range(len(X_test)):
        x1, x2 = X_test[i]

        # Plot true label
        if y_test[i] == 0:
            if not true_0_plotted:
                plt.scatter(x1, x2, color='blue', marker='o', label='True Class 0')
                true_0_plotted = True
            else:
                plt.scatter(x1, x2, color='blue', marker='o')
        else:
            if not true_1_plotted:
                plt.scatter(x1, x2, color='green', marker='o', label='True Class 1')
                true_1_plotted = True
            else:
                plt.scatter(x1, x2, color='green', marker='o')

        # Plot prediction
        prediction = model.predict(X_test[i])
        if not predicted_plotted:
            plt.scatter(x1, x2, color='red', marker='x', label='Predicted')
            predicted_plotted = True
        else:
            plt.scatter(x1, x2, color='red', marker='x')

    # Draw decision boundary
    w = model.weights
    if w[2] != 0:
        x_vals = [0, 1]  # Just need two points to draw a line
        y_vals = [-(w[0] + w[1] * x) / w[2] for x in x_vals]
        plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

    plt.title("Task 1-i-b: Test Predictions vs True Labels")
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
    # Use a larger figure size to show all axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Track if we've added each label to the legend yet
    class0_plotted = False
    class1_plotted = False

    for i in range(len(X_train)):
        X1 = X_train[i][0]
        X2 = X_train[i][1]
        X3 = X_train[i][2]
        label = y_train[i]

        if label == 0:
            color = 'blue'
            marker = 'o'
            legend_label = 'Class 0' if not class0_plotted else None
            class0_plotted = True
        else:
            color = 'red'
            marker = 'x'
            legend_label = 'Class 1' if not class1_plotted else None
            class1_plotted = True

        ax.scatter(X1, X2, X3, color=color, marker=marker, label=legend_label)

    ax.set_title("1-ii-a: 3D Training Data")
    ax.set_xlabel("X1 (Feature 1)")
    ax.set_ylabel("X2 (Feature 2)")
    ax.set_zlabel("X3 (Feature 3)")
    ax.legend()

    # Set a better viewing angle
    ax.view_init(elev=25, azim=120)

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
    # Track if we've added each label to the legend yet
    class0_plotted = False
    class1_plotted = False

    for i in range(len(X_train)):
        X1 = X_train[i][0]
        X2 = X_train[i][1]
        X3 = X_train[i][2]
        label = y_train[i]

        if label == 0:
            color = 'blue'
            marker = 'o'
            legend_label = 'Class 0' if not class0_plotted else None
            class0_plotted = True
        else:
            color = 'red'
            marker = 'x'
            legend_label = 'Class 1' if not class1_plotted else None
            class1_plotted = True

        ax.scatter(X1, X2, X3, color=color, marker=marker, label=legend_label)

    #
    # for i in range (len(X_train)):
    #     X1 = X_train[i][0]
    #     X2 = X_train[i][1]
    #     X3 = X_train[i][2]
    #
    #     predicted_label = model.predict(X_train[i])
    #
    #     if predicted_label == 0:
    #         point_color = 'blue'
    #         point_marker = 'o'
    #         point_label = 'Predicted Class 0' if i == 0 else ''
    #     else:
    #         point_color = 'red'
    #         point_marker = 'x'
    #         point_label = 'Predicted Class 1' if i == 0 else ''
    #
    #     ax.scatter(X1, X2, X3, color=point_color, marker=point_marker, label=point_label)

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
        ax.scatter(X1, X2, X3, color='blue', marker='o', label=point_label)

        # Plot the predicted label (red cross)
        if predicted_label == 0:
            point_label = 'Predicted 0' if i == 0 else ''
        else:
            point_label = 'Predicted 1' if i == 0 else ''
        ax.scatter(X1, X2, X3, color='red', marker='x', label=point_label)

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

# ======================== 2D Graph Wrapper ========================
def show_all_2D_graphs(model, X_train, y_train, X_test, y_test):
    
    print("Displaying 2D Training and Recall Visualizations")
    plot_training_2D_data(X_train, y_train)              # Graph 1-i-a
    plot_decision_boundary_2D(model, X_train, y_train)   # Graph 2-i-a
    plot_training_outputs(model, X_train, y_train)       # Graph 3-i-a
    plot_test_predictions_2D(model, X_test, y_test)      # Graph 1-i-b


# ======================== 3D Graph Wrapper ========================
def show_all_3D_graphs(model, X_train, y_train, X_test, y_test):
    
    print("Displaying 3D Training and Recall Visualizations")
    plot_training_data_3D(X_train, y_train)              # Graph 1-ii-a
    plot_decision_boundary_3D(model, X_train, y_train)   # Graph 2-ii-a
    plot_training_outputs(model, X_train, y_train)       # Graph 3-ii-a
    plot_test_predictions_3D(model, X_test, y_test)      # Graph 1-ii-b





