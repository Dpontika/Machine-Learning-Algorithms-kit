"""
One-vs-Rest (OvR) strategy:

    Train 3 separate Perceptrons:

        Model 1: Classify class 0 vs others

        Model 2: Classify class 1 vs others

        Model 3: Classify class 2 vs others

    Then during prediction, we’ll run all 3 models and take the one with the strongest score or first activation.  

"""

from src.models.perceptron import Perceptron
import numpy as np

class PerceptronOvR:
    
    def __init__(self, input_dim, num_classes, activation='step'):
        """
        Initialize one Perceptron per class.

        Parameters:
        - input_dim: number of input features (e.g., 4 for Iris)
        - num_classes: total number of classes (e.g., 3 for Iris)
        - activation: activation function for each binary Perceptron
        
        """
        self.num_classes = num_classes
        self.models = []
        
        # Create and store one binary Perceptron for each class
        for i in range(num_classes):
            
            print(f"Creating Perceptron model for class {i}")            
            model = Perceptron(input_dim=input_dim, activation=activation)
            self.models.append(model)
        
    def train(self, X_train, y_train, epochs=100, lr=0.1):
        """
        Train one binary Perceptron per class.

        For each class:
        - Convert labels to binary (1 if this class, else 0)
        - Train the corresponding Perceptron
        
        """
        
        for i in range(self.num_classes):
            
            print(f"\Training model for class {i} vs rest")
            #Covert labels to binary
            binary_labels=[]
            for label in y_train:
                if label == i:
                    binary_labels.append(1)
                else:
                    binary_labels.append(0)
            
            self.models[i].train(X_train, binary_labels, epochs=epochs, lr=lr)
    
    def predict(self, x):
        """
        Predict the class label for a single input sample.

        - Calls each binary Perceptron
        - Returns the index (class) of the one with the highest output

        Parameters:
        - x: A single input sample (list or array of features)

        Returns:
        - An integer: predicted class label
        """
        outputs=[]
        
        for i,model in enumerate(self.models):
            prediction = model.predict(x)
            outputs = outputs.append(prediction)
            
        # Return the index of the highest prediction
        return int(np.argmax(outputs))
            
    def predict_batch(self, X):
        """
        Predict class labels for a list of input samples.

        Parameters:
        - X: List or array of input samples (each is a feature vector)

        Returns:
        - A list of predicted class labels (e.g., [0, 2, 1, 1, 0])
        """
        predictions=[]
        
        for i in X:
            predicted_class=self.predict(x)
            predictions.append(predicted_class)
            
        return predictions
            
                        

         

