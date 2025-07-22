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
            # Validate file path
            if not file_path or not file_path.strip():
                raise ValueError("File path cannot be empty")
            
            # Read csv and get column names
            self._data = pd.read_csv(file_path)
            
            # Validate loaded data
            if self._data.empty:
                raise ValueError("CSV file is empty")
            
            self._headers = list(self._data.columns)
            return True
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return False
        except pd.errors.EmptyDataError:
            print(f"File is empty: {file_path}")
            return False
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file: {e}")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
        
    def get_data(self) -> pd.DataFrame:
        """Return copy of loaded data"""
        return self._data.copy() if self._data is not None else None
    
    def get_headers(self) -> List[str]:
        """Return copy of column headers"""
        return self._headers.copy() if self._headers is not None else None
    
    def split_target(self, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Split features from target column"""
        if self._data is None:
            raise ValueError("No data has been loaded.")
        
        # Separate features and target variable
        features = self._data.drop(columns=[target_column])
        target = self._data[target_column]
        return features, target
    