import pandas as pd
from .model import NaiveBayesModel
from .cleaner import Cleaner

class NaiveBayesTrainer:
    """Handles training of Naive Bayes and returns a NaiveBayesModel"""
    def __init__(self, cleaner: Cleaner = None):
        self.cleaner = cleaner if cleaner is not None else Cleaner()

    def train(self, x: pd.DataFrame, y: pd.Series) -> NaiveBayesModel:
        """Train the Naive Bayes model"""
        features = list(x.columns)
        classes = y.unique()
        class_priors = {}
        feature_probabilities = {}
        total_samples = len(y)
        laplace_alpha = self.cleaner.get_laplace_alpha()
        # Calculate prior probabilities for each class
        for class_value in classes:
            class_count = (y == class_value).sum()
            class_priors[class_value] = (class_count + laplace_alpha) / (total_samples + laplace_alpha * len(classes))
        # Calculate conditional probabilities for each feature given each class
        for feature in features:
            feature_probabilities[feature] = {}
            unique_values = x[feature].unique()
            for class_value in classes:
                # Initialize the feature probabilities for the class
                feature_probabilities[feature][class_value] = {}
                class_mask = y == class_value
                class_feature_data = x.loc[class_mask, feature]
                class_size = len(class_feature_data)
                for value in unique_values:
                    value_count = (class_feature_data == value).sum()
                    feature_probabilities[feature][class_value][value] = (value_count + laplace_alpha) / (class_size + laplace_alpha * len(unique_values))
        # Return the model
        return NaiveBayesModel(class_priors, feature_probabilities, classes, features) 