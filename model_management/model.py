from typing import Dict

class NaiveBayesModel:
    """Holds trained parameters for Naive Bayes"""
    def __init__(self, class_priors, feature_probabilities, classes, features):
        self._class_priors = class_priors
        self._feature_probabilities = feature_probabilities
        self._classes = classes
        self._features = features
        self._is_trained = True

    def is_trained(self) -> bool:
        return self._is_trained

    def get_model_info(self) -> Dict:
        return {
            'Status': 'Trained',
            'Classes': list(self._classes),
            'Features': self._features,
            'Number of Classes': len(self._classes),
            'Number of Features': len(self._features)
        }

    @property
    def class_priors(self):
        return self._class_priors

    @property
    def feature_probabilities(self):
        return self._feature_probabilities

    @property
    def classes(self):
        return self._classes

    @property
    def features(self):
        return self._features 