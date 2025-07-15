from src.utils.iris_loader import load_iris_data
from src.models.perceptron_ovr import PerceptronOvR
from src.visualization.iris_multiclass_plots import (
    plot_true_classes,
    plot_predicted_classes,
    plot_perceptron_outputs
)

def run_task_1_iii_a():
    """
    Task 1-iii-a: Multiclass classification using Perceptron (One-vs-Rest)
    This function loads the Iris dataset, trains a custom OvR Perceptron model,
    and shows all 5 required graphs (true labels, predictions, and binary outputs).
    """
    
    print("\n=== Loading Iris dataset ===")
    X_train, X_test, y_train, y_test = load_iris_data()

    print("\n=== Training PerceptronOvR on Iris ===")
    model = PerceptronOvR(input_dim=4, num_classes=3)  # Iris has 4 features
    model.train(X_train, y_train, epochs=20, lr=0.1)

    # Make predictions for all training samples one by one
    y_pred_train = [model.predict(x) for x in X_train]  

    print("\n=== Showing Graph 1: True Classes ===")
    plot_true_classes(X_train, y_train)

    print("\n=== Showing Graph 2: Predicted Classes ===")
    plot_predicted_classes(X_train, y_pred_train)

    print("\n=== Showing Graphs 3–5: Perceptron Outputs (OvR) ===")
    plot_perceptron_outputs(model, X_train, y_train)


# === Entry point ===
if __name__ == "__main__":
    run_task_1_iii_a()