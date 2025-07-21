import pandas as pd
from .naive_bayes_model import NaiveBayesModel

LAPLACE_SMOOTHING_ALPHA = 1.0  # Laplace smoothing constant

class NaiveBayesTrainer:
    """Handles training of Naive Bayes and returns a NaiveBayesModel"""
    def train(self, x: pd.DataFrame, y: pd.Series) -> NaiveBayesModel:
        features = list(x.columns)
        classes = y.unique()
        class_priors = {}
        feature_probabilities = {}
        total_samples = len(y)
        # Calculate prior probabilities for each class
        for class_value in classes:
            class_count = (y == class_value).sum()
            class_priors[class_value] = (class_count + LAPLACE_SMOOTHING_ALPHA) / (total_samples + LAPLACE_SMOOTHING_ALPHA * len(classes))
        # Calculate conditional probabilities for each feature given each class
        for feature in features:
            feature_probabilities[feature] = {}
            unique_values = x[feature].unique()
            for class_value in classes:
                feature_probabilities[feature][class_value] = {}
                class_mask = y == class_value
                class_feature_data = x.loc[class_mask, feature]
                class_size = len(class_feature_data)
                for value in unique_values:
                    value_count = (class_feature_data == value).sum()
                    feature_probabilities[feature][class_value][value] = (value_count + LAPLACE_SMOOTHING_ALPHA) / (class_size + LAPLACE_SMOOTHING_ALPHA * len(unique_values))
        return NaiveBayesModel(class_priors, feature_probabilities, classes, features) 