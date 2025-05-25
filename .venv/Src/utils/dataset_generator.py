"""
This module generates four distinct 2D datasets and two 3D dataset with different separation characteristics:
   Linearly separable (Case a) for 2D & 3D dataset
   Non-linear (angle) (Case b) for 2D dataset
   Non-linear (center) (Case c) for 2D dataset
   Non-linear (XOR) (Case d) or 2D & 3D dataset
   Each function returns (X_train, X_test, y_train, y_test) for immediate use in training/testing.

"""

import numpy as np
from sklearn.model_selection import train_test_split

#Case a: Linearly separable 2D data
def generate_2D_linear_separable(n=100):
    np.random.seed(42)

    #Class 0: Bottom-left quadrant x-range: [0.0, 0.3] y-range: [0.0, 0.3]
    class0 = np.random.uniform(low=[0.0, 0.0], high=[0.3, 0.3], size=(n//2,2))
    #Class 1: Top-right quadrant x-range: [0.7, 0.9] y-range: [0.7, 0.9]
    class1 = np.random.uniform(low=[0.7, 0.7], high=[0.9, 0.9], size=(n//2,2))

    #Vertically stacks the arrays class0 and class1 to create a single array X containing all the data points
    X=np.vstack((class0, class1))
    # The first n//2 elements as zeros (representing Class 0).
    # The next n//2 elements as ones (representing Class 1).
    y=np.array([0]*(n//2) + [1]*(n//2))

    return train_test_split(X, y, test_size=0.2, random_state=42)

#Linearly seperable 3D data
def generate_3D_linear_separable(n=100):
    """
        Create a linearly separable 3D dataset:
          - Class 0: n/2 points uniformly in [0.0,0.3]^3
          - Class 1: n/2 points uniformly in [0.7,0.9]^3
        Returns train/test splits.
        """
    # Fix the random seed so results are reproducible
    np.random.seed(42)

    half = n // 2

    # Class 0 in the “lower corner”
    class0 = np.random.uniform(
        low=[0.0, 0.0, 0.0],
        high=[0.3, 0.3, 0.3],
        size=(half, 3)
    )

    # Class 1 in the “upper corner”
    class1 = np.random.uniform(
        low=[0.7, 0.7, 0.7],
        high=[0.9, 0.9, 0.9],
        size=(half, 3)
    )

    # Stack features and labels
    X = np.vstack((class0, class1))
    y = np.array([0] * half + [1] * half)

    # Split into train/test
    return train_test_split(X, y, test_size=0.2, random_state=42)

#Case b: Non-linear 2D (angle)
def generate_2D_nonlinear_angle(n=100):
    np.random.seed(42)

    #Class 0: Bottom-left quadrant x-range: [0.0, 0.3] y-range: [0.0, 0.3]
    class0 = np.random.uniform(low=[0.0, 0.0], high=[0.3, 0.3], size=(n // 2, 2))

    # Class 1: Top-left and right half
    #n//4 points in: x: [0.0, 0.3] - y: [0.4, 0.9]
    class1_part1 = np.random.uniform(low=[0.0, 0.4], high=[0.3, 0.9], size=(n // 4, 2))
    # n//4 points in: x: [0.4, 0.9] - y: [0.0, 0.9]
    class1_part2 = np.random.uniform(low=[0.4, 0.0], high=[0.9, 0.9], size=(n // 4, 2))

    X = np.vstack((class0, class1_part1, class1_part2))
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    return train_test_split(X, y, test_size=0.2, random_state=42)

#Case c: Non-linear 2D (center). This creates a non-linearly separable pattern where class 0 is completely surrounded by class 1.
def generate_2D_nonlinear_center(n=100):
 np.random.seed(42)

 # Class 0: Center square [0.4-0.6] x [0.4-0.6]
 class0 = np.random.uniform(low=[0.4, 0.4], high=[0.6, 0.6], size=(n // 2, 2))

 # Class 1: Surrounding frame (4 regions)
 class1_part1 = np.random.uniform(low=[0.0, 0.0], high=[0.9, 0.3], size=(n // 8, 2))  # Bottom
 class1_part2 = np.random.uniform(low=[0.0, 0.7], high=[0.9, 0.9], size=(n // 8, 2))  # Top
 class1_part3 = np.random.uniform(low=[0.0, 0.0], high=[0.3, 0.9], size=(n // 8, 2))  # Left
 class1_part4 = np.random.uniform(low=[0.7, 0.0], high=[0.9, 0.9], size=(n // 8, 2))  # Right

 X = np.vstack((class0, class1_part1, class1_part2, class1_part3, class1_part4))
 y = np.array([0] * (n // 2) + [1] * (n // 2))

 return train_test_split(X, y, test_size=0.2, random_state=42)

#Case d: Non-linear 2D (XOR)
def generate_2D_nonlinear_xor(n=100):
 # Class 0: Bottom-left + Top-right
 class0_part1 = np.random.uniform(low=[0.0, 0.0], high=[0.3, 0.3], size=(n // 4, 2))
 class0_part2 = np.random.uniform(low=[0.7, 0.7], high=[0.9, 0.9], size=(n // 4, 2))

 # Class 1: Top-left + Bottom-right
 class1_part1 = np.random.uniform(low=[0.7, 0.0], high=[0.9, 0.3], size=(n // 4, 2))
 class1_part2 = np.random.uniform(low=[0.0, 0.7], high=[0.3, 0.9], size=(n // 4, 2))

 X = np.vstack((class0_part1, class0_part2, class1_part1, class1_part2))
 y = np.array([0] * (n // 2) + [1] * (n // 2))

 return train_test_split(X, y, test_size=0.2, random_state=42)

 #Non-linear 3D (XOR)

def generate_3D_nonlinear_xor(n=100):
 np.random.seed(42)

 # Class 0: cube at origin corner and opposite corner
 class0_part1 = np.random.uniform(low=[0.0, 0.0, 0.0], high=[0.3, 0.3, 0.3], size=(n // 4, 3))
 class0_part2 = np.random.uniform(low=[0.7, 0.7, 0.7], high=[0.9, 0.9, 0.9], size=(n // 4, 3))
 # Class 1: XOR cubes
 class1_part1 = np.random.uniform(low=[0.7, 0.7, 0.0], high=[0.9, 0.9, 0.3], size=(n // 4, 3))
 class1_part2 = np.random.uniform(low=[0.0, 0.0, 0.7], high=[0.3, 0.3, 0.9], size=(n // 4, 3))

 X = np.vstack((class0_part1, class0_part2, class1_part1, class1_part2))
 y = np.array([0] * (n // 2) + [1] * (n // 2))

 return train_test_split(X, y, test_size=0.2, random_state=42)










