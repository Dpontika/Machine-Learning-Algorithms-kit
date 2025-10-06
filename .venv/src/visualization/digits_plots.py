import matplotlib.pyplot as plt
import numpy as np

def plot_training_outputs_digits(y_true, y_pred):
    """
    Plots true outputs (target) and predicted outputs for each class (digit).
    Each class (5,6,8,9) is represented as on graph:
        - Blue dots = true targets (what the digit really is)
        - Red circles = model predictions (what the model thinks)
        
    Parameters: 
        - y_true: true class labels (0, 1, 2, 3)
        - y_pred: predicted class labels (0, 1, 2, 3)
        
    """
    # Create sample indices for the x-axis
    sample_indices = list(range(len(y_true)))
    
    # Create plot
    plt.figure(figsize=(10,5))
    plt.scatter(sample_indices, y_true, color='blue', marker='o', label='True Labels')
    plt.scatter(sample_indices, y_pred, color='red', marker='x', label = 'Predicted Labels')
    
    plt.title("Task 1-v-a: Perceptron Outputs vs True Targets (Training Data)")
    plt.xlabel("Sample Index")
    plt.ylabel("Class Labels (0=5, 1=6, 2=8, 3=9)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


    
    