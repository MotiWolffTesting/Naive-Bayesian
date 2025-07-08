import pandas as pd
from .naive_bayes_classifier import NaiveBayesClassifier
from typing import Dict, Any

class ClassificationEngine:
    """Classification Engine wrapper for Naive Bayes model"""
    def __init__(self):
        # Initialize the classifier
        self._classifier = NaiveBayesClassifier()
        # Store target column name for consistency
        self._target_column = None
        
    def build_model(self, data: pd.DataFrame, target_column: str) -> bool:
        """Build and train the classification model"""
        try:
            # Validate target column exists
            if target_column not in data.columns:
                print("Target column not found in the data.")
                return False
    
            # Store target column and split data
            self._target_column = target_column
            x = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Train the classifier
            self._classifier.train(x, y)
            return True
        except Exception as e:
            print(f"Error building model: {e}.")
            return False
        
    def classify_single_record(self, record: Dict[str, Any]) -> str:
        """Classify a single record and return predicted class"""
        return self._classifier.predict_single(record)
    
    def test_model_accuracy(self, test_data: pd.DataFrame) -> float:
        """Test model accuracy on test dataset"""
        if self._target_column not in test_data.columns:
            raise ValueError("Test data must contain the target column")
        
        # Split test data into features and target
        x_test = test_data.drop(columns=[self._target_column])
        y_test = test_data[self._target_column]
        
        # Get predictions for all test samples
        predictions = self._classifier.predict_group(x_test)
        # Count correct predictions
        correct_predictions = sum(1 for prediction, actual in zip(predictions, y_test) if prediction == actual)
        accuracy = correct_predictions / len(y_test)
        
        # Display results
        print("Test Results:")
        print(f"Total Records: {len(y_test)}.")
        print(f"Correct Classifications: {correct_predictions}.")
        print(f"Model Accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def get_classifier_info(self) -> Dict:
        """Return information about the underlying classifier"""
        return self._classifier.get_model_info()
    
    def is_model_ready(self) -> bool:
        """Check if the model is trained and ready for predictions"""
        return self._classifier.is_trained()
    
    
        
        
        