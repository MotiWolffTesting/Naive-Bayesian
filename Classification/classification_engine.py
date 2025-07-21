import pandas as pd
from .naive_bayes_trainer import NaiveBayesTrainer
from .naive_bayes_classifier import NaiveBayesClassifier
from .naive_bayes_model import NaiveBayesModel
from typing import Dict, Any

class ClassificationEngine:
    """Classification Engine wrapper for Naive Bayes model"""
    def __init__(self):
        self._trainer = NaiveBayesTrainer()
        self._model = None
        self._classifier = None
        self._target_column = None
        
    def build_model(self, data: pd.DataFrame, target_column: str) -> bool:
        """Build and train the classification model"""
        try:
            if data is None or data.empty:
                raise ValueError("Data cannot be None or empty")
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the data")
            self._target_column = target_column
            x = data.drop(columns=[target_column])
            y = data[target_column]
            if x.empty:
                raise ValueError("No features available for training")
            self._model = self._trainer.train(x, y)
            self._classifier = NaiveBayesClassifier(self._model)
            return True
        except Exception as e:
            print(f"Error building model: {e}")
            return False
        
    def classify_single_record(self, record: Dict[str, Any]) -> str:
        """Classify a single record and return predicted class"""
        if not self._classifier:
            raise ValueError("Model is not trained yet.")
        return self._classifier.classify_single(record)
    
    def test_model_accuracy(self, test_data: pd.DataFrame, target_column: str = None) -> float:
        """Test model accuracy on test dataset"""
        test_target_column = target_column if target_column else self._target_column
        if test_target_column not in test_data.columns:
            raise ValueError(f"Test data must contain the target column '{test_target_column}'")
        x_test = test_data.drop(columns=[test_target_column])
        y_test = test_data[test_target_column]
        if not self._classifier:
            raise ValueError("Model is not trained yet.")
        predictions = self._classifier.classify_group(x_test)
        correct_predictions = sum(1 for prediction, actual in zip(predictions, y_test) if prediction == actual)
        accuracy = correct_predictions / len(y_test)
        print("Test Results:")
        print(f"Total Records: {len(y_test)}.")
        print(f"Correct Classifications: {correct_predictions}.")
        print(f"Model Accuracy: {accuracy:.2%}")
        return accuracy
    
    def get_classifier_info(self) -> Dict:
        """Return information about the underlying classifier"""
        if not self._model:
            return {'Status': 'Not trained'}
        return self._model.get_model_info()
    
    def is_model_ready(self) -> bool:
        """Check if the model is trained and ready for predictions"""
        return self._model is not None and self._model.is_trained()
    
    
        
        
        