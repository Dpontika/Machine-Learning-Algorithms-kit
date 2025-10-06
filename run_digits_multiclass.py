"""
Task 1-v-a: Multiclass Perceptron – Digit Classification (5, 6, 8, 9)
This script:
Loads digit data (perfect + imperfect samples)
Trains a multiclass Perceptron (OvR)
Plots true vs predicted outputs for training data (Graphs 1–4)
"""


from src.utils.digits_loader import load_digits_dataset
from src.models.perceptron_ovr import PerceptronOvR
from src.visualization.digits_plots import plot_training_outputs_digits


def run_task_1_v_a():
    print("\n=== Task 1-v-a: Digit Classification (5,6,8,9) – Training Phase ===")
    
    print("\n=== Loading Digits dataset ===")
    X_train, X_test, y_train, y_test = load_digits_dataset()
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
 
    print("\n=== Training PerceptronOvR on Digits ===")
    # Digits 5,6,8,9 → 4 classes
    model = PerceptronOvR(input_dim=X_train.shape[1], num_classes=4, activation='step') # (64 for 8x8 digit images)
    model.train(X_train, y_train, epochs=20, lr=0.1)

    # Prediction
    y_pred_train = model.predict_batch(X_train)
    
    print("\n=== Showing Graphs 1–4: True vs Predicted Outputs (Training Data) ===")
    plot_training_outputs_digits(y_train, y_pred_train)
    

#=== Entry Point ===
if __name__ == "__main__":
    run_task_1_v_a()

