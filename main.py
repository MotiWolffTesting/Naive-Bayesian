from Classification.classification_engine import ClassificationEngine
from utils.data_loader import DataLoader
from UI.console_interface import ConsoleInterface
import os

class Main:
    """Main application class for Naive Bayes Classifier"""
    def __init__(self):
        # Initialize data processing components
        self._data_loader = DataLoader()
        self._classification_engine = ClassificationEngine()
        self._ui = ConsoleInterface()  # Fix: Added missing parentheses
        self._training_data = None
        
    def run(self):
        """Start the application workflow"""
        self._ui.display_message("=== Naive Bayes Classifier")
        
        # Step 1: Load training data
        if not self._load_training_data():
            return
        
        # Step 2: Build model
        if not self._build_model():
            return
        
        # Step 3: Show main menu
        self._main_menu()
        
        
    def _load_training_data(self) -> bool:
        """Load training data from CSV file"""
        while True:
            file_path = self._ui.get_user_input("Enter file path (csv): ")
            
            if not os.path.exists(file_path):
                self._ui.display_message("File not found, try again.")
                continue
            
            if self._data_loader.load_csv(file_path=file_path):
                self._training_data = self._data_loader.get_data()
                self._ui.display_message(f"{len(self._training_data)} records have been loaded successfully.")
                self._ui.display_message(f"Columns: {', '.join(self._data_loader.get_headers())}")
                return True
            else:
                retry = self._ui.get_user_input("Should I try again? (yes / no): ")
                if retry.lower() != "yes":
                    return False
    
    def _build_model(self) -> bool:
        """Build and train the classification model"""
        headers = self._data_loader.get_headers()
        self._ui.display_message("Available Columns: ")
        # Display numbered column options
        for i, header in enumerate(headers):
            self._ui.display_message(f"{i+1}. {header}")
            
        while True:
            target_col = self._ui.get_user_input("Enter target column name: ")
            
            if target_col in headers:
                if self._classification_engine.build_model(self._training_data, target_col):
                    self._ui.display_message("Model built successfully!")
                    
                    # Display model information
                    model_info = self._classification_engine.get_classifier_info()
                    self._ui.display_message(f"Classes number: {model_info['Number of Classes']}")
                    self._ui.display_message(f"Classes: {model_info['Classes']}")
                    self._ui.display_message(f"Features number: {model_info['Number of Features']}")
                    
                    return True
                else:
                    return False
                
            else:
                self._ui.display_message("Column not exists, try again")
    
    def _main_menu(self):
        """Display and handle main menu options"""
        options = [
            "Check model accuracy with a data file",
            "Classify a single record",
            "Display info on model",
            "Exit"
        ]
        
        while True:
            choice = self._ui.get_menu_choice(options)
            
            if choice == 0:
                self._test_model_accuracy()
            elif choice == 1:
                self._classify_single_record()
            elif choice == 2:
                self._show_model_info()
            elif choice == 3:  # Fix: Changed from 4 to 3 (0-based indexing)
                self._ui.display_message("Thank you for using the classifier. Good Bye!")
                break
            
    def _test_model_accuracy(self):
        """Test model accuracy on external dataset"""
        file_path = self._ui.get_user_input("Enter a file path: ")
        
        if not os.path.exists(file_path):
            self._ui.display_message("File not found.")
            return
        
        test_loader = DataLoader()
        if test_loader.load_csv(file_path=file_path):
            test_data = test_loader.get_data()
            try:
                # Call test method (accuracy is displayed internally)
                self._classification_engine.test_model_accuracy(test_data)
            except Exception as e:
                self._ui.display_message(f"Error testing accuracy: {e}.")
                
        else:
            self._ui.display_message("Error loading test file.")
            
    def _classify_single_record(self):
        """Classify a single user-input record"""
        model_info = self._classification_engine.get_classifier_info()
        features = model_info['Features']
        
        self._ui.display_message("Enter values for the following features: ")
        
        # Collect feature values from user
        record = {}
        for feature in features:
            value = self._ui.get_user_input(f"{feature}: ")
            record[feature] = value
            
        try:
            result = self._classification_engine.classify_single_record(record=record)
            self._ui.display_message(f"Classification results: {result}.")
        except Exception as e:
            self._ui.display_message(f"Error classifying: {e}.")
            
        
    def _show_model_info(self):
        """Display detailed model information"""
        info = self._classification_engine.get_classifier_info()
        self._ui.display_message("\nModel Information")
        # Format and display each piece of model information
        for key, value in info.items():
            if key == "Features":
                self._ui.display_message(f"{key}: {', '.join(value)}")
            elif key == 'Classes':
                self._ui.display_message(f"{key}: {', '.join(map(str, value))}")
            else:
                self._ui.display_message(f"{key}: {value}")

if __name__ == "__main__":
    # Create and run the main application
    app = Main()
    app.run()