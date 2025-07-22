import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from typing import Tuple, Any

class Validator:
    """Provides validation utilities such as train-test split and confusion matrix."""
    def split_data(self, data: pd.DataFrame, target_column: str, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets (default 70/30)."""
        x = data.drop(columns=[target_column])
        y = data[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
        return x_train, x_test, y_train, y_test

    def compute_confusion_matrix(self, y_true: Any, y_pred: Any) -> Any:
        """Compute confusion matrix given true and predicted labels."""
        return confusion_matrix(y_true, y_pred)
