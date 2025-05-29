# Import dataset generation functions
from Src.utils.dataset_generator import generate_2D_linear_separable, generate_3D_linear_separable

# Import your custom Perceptron model
from Src.models.perceptron import Perceptron

# Import all the plotting functions for 2D and 3D tasks
from Src.visualization.perceptron_plots import (
    plot_training_2D_data,
    plot_decision_boundary_2D,
    plot_training_outputs,
    plot_test_predictions_2D,
    plot_errors_per_epoch,
    plot_training_data_3D,
    plot_decision_boundary_3D,
    plot_test_predictions_3D
)


def run_task_1_i():  # 1-i-a and 1-i-b (2D classification)
    print("\n=== Task 1-i-a: 2D Training Phase ===")

    # Generate 2D dataset (linearly separable)
    X_train, X_test, y_train, y_test = generate_2D_linear_separable(n=100)

    # Create a Perceptron model for 2 input features
    model = Perceptron(input_dim=2, activation='step')

    # Train the model on the training data
    errors = model.train(X_train, y_train, epochs=20, lr=0.1)

    # Show 2D training results
    plot_training_2D_data(X_train, y_train)              # Graph 1
    plot_decision_boundary_2D(model, X_train, y_train)   # Graph 2
    plot_training_outputs(model, X_train, y_train)       # Graph 3
    plot_errors_per_epoch(errors)                        # Training progress

    print("\n=== Task 1-i-b: 2D Recall Phase ===")
    plot_test_predictions_2D(model, X_test, y_test)      # Test performance


def run_task_1_ii():  # 1-ii-a and 1-ii-b (3D classification)
    print("\n=== Task 1-ii-a: 3D Training Phase ===")

    # Generate a 3D dataset (linearly separable)
    X_train, X_test, y_train, y_test = generate_3D_linear_separable(n=100)

    # Create a Perceptron model for 3 input features
    model = Perceptron(input_dim=3, activation='step')

    # Train the model on the training data
    model.train(X_train, y_train, epochs=20, lr=0.1)

    # Show 3D training results
    plot_training_data_3D(X_train, y_train)             # Graph 1
    plot_decision_boundary_3D(model, X_train, y_train)  # Graph 2 (plane)
    plot_training_outputs(model, X_train, y_train)      # Graph 3

    print("\n=== Task 1-ii-b: 3D Recall Phase ===")
    plot_test_predictions_3D(model, X_test, y_test)     # Test performance
    


# Run script
if __name__ == "__main__":
    run_task_1_i()
    run_task_1_ii()
