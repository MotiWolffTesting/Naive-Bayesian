import pandas as pd
from typing import List, Dict, Any
import numpy as np

class NaiveBayesClassifier:
    """Naive Bayesian Classifier with Laplace smoothing"""
    def __init__(self):
        # Store prior probabilities for each class
        self._class_priors = {}
        # Store conditional probabilities for each feature-class combination
        self._feature_probabilities = {}
        # Track unique classes and features
        self._classes = None
        self._features = None
        # Training status flag
        self._is_trained = False
        
    def train(self, x: pd.DataFrame, y: pd.Series):
        """Train the model on features and target values"""
        # Store feature names and unique classes
        self._features = list(x.columns)
        self._classes = y.unique()
        
        # Calculate prior probabilities for each class
        total_samples = len(y)
        for class_value in self._classes:
            class_count = (y == class_value).sum()
            # Apply Laplace smoothing: (count + 1) / (total + num_classes)
            self._class_priors[class_value] = (class_count + 1) / (total_samples + len(self._classes))
        
        # Calculate conditional probabilities for each feature given each class
        self._feature_probabilities = {}
        for feature in self._features:
            self._feature_probabilities[feature] = {}
            unique_values = x[feature].unique()
            
            for class_value in self._classes:
                self._feature_probabilities[feature][class_value] = {}
                # Get data for this class only
                class_mask = y == class_value
                class_feature_data = x.loc[class_mask, feature]
                class_size = len(class_feature_data)
                
                for value in unique_values:
                    value_count = (class_feature_data == value).sum()
                    # Apply Laplace smoothing for conditional probabilities
                    self._feature_probabilities[feature][class_value][value] = (value_count + 1) / (class_size + len(unique_values))
        
        # Mark model as trained
        self._is_trained = True
        
    
    def predict_single(self, sample: Dict[str, Any]) -> str:
        """Classify a single sample using Naive Bayes"""
        if not self._is_trained:
            raise ValueError("Model has not been trained yet.")
        
        classes_scores = {}
        
        for class_value in self._classes:
            # Start with log of prior probability
            log_probability = np.log(self._class_priors[class_value])
            
            # Add log probabilities for each feature
            for feature, value in sample.items():
                if feature in self._feature_probabilities:
                    if value in self._feature_probabilities[feature][class_value]:
                        feature_probability = self._feature_probabilities[feature][class_value][value]
                    else:
                        # Handle unseen values with Laplace smoothing
                        num_unique = len(self._feature_probabilities[feature][class_value])
                        class_size = sum(self._feature_probabilities[feature][class_value].values()) * (num_unique - 1)
                        feature_probability = 1 / (class_size + num_unique + 1)
                        
                    log_probability += np.log(feature_probability)
                
            classes_scores[class_value] = log_probability
        
        # Return class with highest probability score
        return max(classes_scores, key=classes_scores.get)
    
    def predict_group(self, x: pd.DataFrame) -> List[str]:
        """Classify multiple samples"""
        predictions = []
        # Iterate through each row and predict
        for _, row in x.iterrows():
            sample = row.to_dict()
            prediction = self.predict_single(sample)
            predictions.append(prediction)
        return predictions
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._is_trained
    
    def get_model_info(self) -> Dict:
        """Return model information and statistics"""
        if not self._is_trained:
            return {'Status': 'Not trained'}
        
        return {
            'Status': 'Trained',
            'Classes': list(self._classes),
            'Features': self._features,
            'Number of Classes': len(self._classes),
            'Number of Features': len(self._features)
        }
        
        
            
    
    
    
        
    
    