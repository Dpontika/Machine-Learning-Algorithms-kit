"""
Multiclass classification (3 classes - Iris dataset) plots in 2D using:

    X-axis: Feature 1 → sepal length

    Y-axis: Feature 3 → petal length

    (Column indices 0 and 2 in the data.)
    
Graph 1: True training samples by class

Graph 2: Predicted training classes 

Graphs 3, 4, 5:	For each class-specific Perceptron, plot its binary outputs over training data (activation: 0/1), 
showing how well it separates its class from the rest

"""
import matplotlib.pyplot as plt 

# Graph 1
def plot_true_classes(X_train, y_train):
    """
    Plot the true class labels of the Iris training data in 2D,
    using sepal length (feature 0) and petal length (feature 2).

    Parameters:
    - X_train: 2D array of shape [n_samples, 4] (full Iris features)
    - y_train: 1D array of shape [n_samples] with labels (0, 1, or 2)
    """
    plt.figure(figsize=(8,6))
    
    # Loop through each class (0 = Setosa, 1 = Versicolor, 2 = Virginica)
    for class_label in [0, 1, 2]:

        # Create two empty lists to hold the x and y coordinates for this class
        x_vals = []  # Feature 0: Sepal length (x-axis)
        y_vals = []  # Feature 2: Petal length (y-axis)

        # Loop through all training samples
        for i in range(len(X_train)):
            # If this sample belongs to the current class, store its features
            if y_train[i] == class_label:
                x_vals.append(X_train[i][0])  # Feature 0 (sepal length)
                y_vals.append(X_train[i][2])  # Feature 2 (petal length)


        if class_label == 0:
            color = 'blue'; marker = 'o'; label_name = 'Setosa (class 0)'
        elif class_label == 1:
            color = 'green'; marker = 's'; label_name = 'Versicolor (class 1)'
        else:
            color = 'red'; marker = '^'; label_name = 'Virginica (class 2)'

        plt.scatter(x_vals, y_vals, color=color, marker=marker, label=label_name)

    plt.title("Graph 1: True Training Classes")
    plt.xlabel("Sepal Length (Feature 0)")
    plt.ylabel("Petal Length (Feature 2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
 
# Graph 2   
def plot_predicted_classes(X_train, y_pred):
    """
    Plot the predicted class labels for the training samples in 2D,
    using sepal length (feature 0) and petal length (feature 2).

    Parameters:
    - X_train: Training features (array or list of shape [n_samples, 4])
    - y_pred: Predicted class labels (0, 1, or 2), one for each sample
    """
    plt.figure(figsize=(8, 6))
    
    for class_label in [0, 1, 2]:
        x_vals = []
        y_vals = []

        for i in range(len(X_train)):
            if y_pred[i] == class_label:
                # Keep only feature 0 (sepal length) and feature 2 (petal length)
                x_vals.append(X_train[i][0])
                y_vals.append(X_train[i][2])

        if class_label == 0:
            color = 'blue'; marker = 'o'; label_name = 'Predicted Setosa (class 0)'
        elif class_label == 1:
            color = 'green'; marker = 's'; label_name = 'Predicted Versicolor (class 1)'
        else:
            color = 'red'; marker = '^'; label_name = 'Predicted Virginica (class 2)'

        plt.scatter(x_vals, y_vals, color=color, marker=marker, label=label_name)

    plt.title("Graph 2: Predicted Training Classes")
    plt.xlabel("Sepal Length (Feature 0)")
    plt.ylabel("Petal Length (Feature 2)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Graph 3,4,5    
def plot_perceptron_outputs(model, X_train, y_train):
    """
    Plot 3 graphs, one for each binary Perceptron in the OvR model.
    Each graph shows whether that Perceptron predicts 0 or 1 for each training sample.

    Parameters:
    - model: Trained PerceptronOvR model (contains 3 binary Perceptrons)
    - X_train: Training data (n_samples x 4)
    - y_train: True labels (0, 1, or 2)
    """
    # 3 subplots (1 row, 3 colunms)
    fig, axs = plt.subplots(1,3,figsize=(18,5))
    
    # Loop through the 3 binary Perceptrons
    for class_index in range(3):
        
        # Select subplot
        ax=axs[class_index]
        
        # For this Perceptron, plot its output (0 or 1) for every training point
        for i in range(len(X_train)):
            x1 = X_train[i][0]
            x2 = X_train[i][2]
            
            prediction=model.models[class_index].predict(X_train[i])
            
            if prediction == 1: color='green'; marker='o'
            else: color='grey'; marker = 'x'
            
            ax.scatter(x1,x2,color=color,marker=marker)
            
        ax.set_title(f"Graph {class_index + 3}: Output for Perceptron class {class_index}")
        ax.set_xlabel("Sepal Length (Feature 0)")
        ax.set_ylabel("Petal Length (Feature 2)")
        ax.grid(True)
        
    plt.tight_layout()
    plt.show()
        
        
    