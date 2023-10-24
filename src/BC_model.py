import numpy as np
import pandas as pd

class BordaCountClassifier:
    
    # Initialize the BordaCountClassifier
    def __init__(self, estimators):
        self.estimators = estimators

    # Fit each estimator to the data
    def fit(self, X, y):
        for _, estimator in self.estimators:
            estimator.fit(X, y)


    # Predict the class labels using the Borda Count approach
    def predict(self, X):
        
        # Get the predicted probabilities for each classifier
        all_probs = [estimator.predict_proba(X) for _, estimator in self.estimators]
        all_probs = np.stack(all_probs)

        # Get rankings for each classifier's predictions
        rankings = np.argsort(-all_probs, axis=-1)

        # Assign points based on rankings
        num_classes = all_probs.shape[2]
        points = np.zeros_like(rankings)
        for rank in range(num_classes):
            points[rankings == rank] = num_classes - 1 - rank

        # Sum points across classifiers
        total_points = points.sum(axis=0)

        # Get the final prediction as the class with the highest total points
        final_predictions = np.argmax(total_points, axis=1)

        return final_predictions