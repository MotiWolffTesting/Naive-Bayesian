from model_management.model import NaiveBayesModel
import numpy as np
from typing import List, Dict, Any

DEFAULT_UNSEEN_PROBABILITY = 1e-10  # Probability for unseen values

class NaiveBayesClassifier:
    """Classifies samples using a trained NaiveBayesModel"""
    def __init__(self, model: NaiveBayesModel):
        if not model.is_trained():
            raise ValueError("Model has not been trained yet.")
        self._model = model

    def classify_single(self, sample: Dict[str, Any]) -> str:
        """Classify a single sample using the trained model"""
        classes_scores = {}
        for class_value in self._model.classes:
            # Calculate the log probability of the class
            log_probability = np.log(self._model.class_priors[class_value])
            for feature, value in sample.items():
                if feature in self._model.feature_probabilities:
                    if value in self._model.feature_probabilities[feature][class_value]:
                        feature_probability = self._model.feature_probabilities[feature][class_value][value]
                    else:
                        feature_probability = DEFAULT_UNSEEN_PROBABILITY
                    log_probability += np.log(feature_probability)
            classes_scores[class_value] = log_probability
        # Return the class with the highest score
        return max(classes_scores, key=classes_scores.get)

    def classify_group(self, x):
        """Classify a group of samples using the trained model"""
        predictions = []
        for _, row in x.iterrows():
            # Convert each row to a dictionary
            sample = row.to_dict()
            prediction = self.classify_single(sample)
            predictions.append(prediction)
        # Return the predictions
        return predictions
        
        
            
    
    
    
        
    
    