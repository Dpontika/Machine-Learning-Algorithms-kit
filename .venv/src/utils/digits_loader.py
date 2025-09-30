import numpy as np
from sklearn.datasets import load_digits

def load_digits_dataset(train_per_class=8, test_per_class=3, random_state=42):
    """
    Load and prepare digit data for classes 5, 6, 8, 9.
    Splits into training and testing sets.

    Parameters:
    - train_per_class: Number of training samples per digit (default=8)
    - test_per_class: Number of testing samples per digit (default=3)
    - random_state: Random seed for reproducibility

    Returns:
    - X_train, X_test: Feature arrays (images flattened into 1D vectors)
    - y_train, y_test: Labels (0=class for digit 5, 1=6, 2=8, 3=9)
    """

    #Load digits dataset (0-9)
    digits = load_digits()
    X, y = digits.data, digits.target   # X = images, y = labels

    print("dataset shape:", X.shape)   
    print("Labels:", np.unique(y))  

    # Select only digits 5, 6, 8, 9
    selected_digits = [5, 6, 8, 9]
    mask = np.isin(y, selected_digits)   # Boolean mask
    X, y = X[mask], y[mask]             # Filtered dataset

    print("Shape after filtering:", X.shape)
    print("Unique labels:", np.unique(y))

    # Map labels to class IDs: 5→0, 6→1, 8→2, 9→3
    label_map = {5: 0, 6: 1, 8: 2, 9: 3}
    y_mapped = np.array([label_map[val] for val in y])
    #print(y_mapped)

    # Split into train/test sets per class
    X_train, y_train, X_test, y_test = [], [], [], []

    np.random.seed(random_state)  # reproducibility

    for digit, class_id in label_map.items():
        # Get indices of all samples of this digit
        digit_indices = np.where(y == digit)[0]

        print(f"\nDigit {digit} → class {class_id}")
        print("Total samples available:", len(digit_indices))

        # Shuffle indices
        np.random.shuffle(digit_indices)

        # Select training and test samples
        train_idx = digit_indices[:train_per_class]
        test_idx = digit_indices[train_per_class:train_per_class + test_per_class]

        print("Training indices:", train_idx)
        print("Testing indices:", test_idx)

        # Add samples to train/test sets
        X_train.extend(X[train_idx])
        y_train.extend([class_id] * train_per_class)

        X_test.extend(X[test_idx])
        y_test.extend([class_id] * test_per_class)

    # Convert to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print("\nFinal dataset sizes:")
    print("Training:", X_train.shape, y_train.shape)
    print("Testing:", X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


# Run for testing/debugging
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_digits_dataset()
