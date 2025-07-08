import pandas as pd
from typing import List, Tuple

class DataLoader:
    """Class for loading and converting csv files"""
    def __init__(self):
        # Store loaded data and column headers
        self._data = None
        self._headers = None
        
    def load_csv(self, file_path: str) -> bool:
        """Load csv file and store data with headers"""
        try:
            # Read csv and get column names
            self._data = pd.read_csv(file_path)
            self._headers = list(self._data.columns)
            return True
        except Exception as e:
            print(f"Error loading file: {e}.")
            return False
        
    def get_data(self) -> pd.DataFrame:
        """Return copy of loaded data"""
        return self._data.copy() if self._data is not None else None
    
    def get_headers(self) -> List[str]:
        """Return copy of column headers"""
        return self._headers.copy() if self._headers is not None else None
    
    def split_target(self, taregt_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Split features from target column"""
        if self._data is None:
            return ValueError("No data has been loaded.")
        
        # Separate features and target variable
        features = self._data.drop(columns=[taregt_column])
        target = self._data[taregt_column]
        return features, target
    