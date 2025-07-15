import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

def load_iris_data(test_size=0.2, random_state=42):
    """
    Loads the Iris dataset from UCI and returns training and test sets.

    Parameters:
    - test_size: Fraction of data to use for testing (default 0.2 --> 20%)
    - random_state: Seed for reproducibility

    Returns:
    - X_train, X_test: Features (sepal/petal size info)
    - y_train, y_test: Labels (0 = setosa, 1 = versicolor, 2 = virginica)
    """
    
    iris = fetch_ucirepo(id=53)
    
    # Get the data and labels
    X = iris.data.features  # DataFrame with 4 columns (features)
    y = iris.data.targets  # DataFrame with the class labels (as strings)

    # Convert class labels (strings) to numbers (0, 1, 2)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values.ravel())

    # Split into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)

    return X_train.values, X_test.values, y_train, y_test

    #return X_train, X_test, y_train, y_test, X, y, y_encoded, label_encoder


# # Info
# X_train, X_test, y_train, y_test, X, y, y_encoded, label_encoder = load_iris_data()

# # Print shapes of X and y
# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)

# # Print the first few rows of X and y
# print("Head of X:\n", X.head())
# print("Head of y:\n", y.head())

# # Print column names of X and value counts of y
# print("Columns of X:", X.columns)
# print("Value counts of y:\n", y.value_counts())

# # Print the first 10 encoded labels and the classes
# print("First 10 encoded labels:", y_encoded[:10])
# print("Classes:", label_encoder.classes_)

# # Print shapes of the training and test sets
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_test shape:", y_test.shape)



    

    