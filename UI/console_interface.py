from UI.user_interface import UserInterface
from typing import List

class ConsoleInterface(UserInterface):
    """Console-based user interface implementation"""
    
    def display_message(self, message: str):
        """Display a message to the console"""
        print(message)
        
    def get_user_input(self, prompt: str) -> str:
        """Get text input from user"""
        return input(prompt)
    
    def get_menu_choice(self, options: List[str]) -> int:
        """Display menu options and get user's choice"""
        print("\nChoose option: ")
        # Display numbered menu options
        for i, option in enumerate(options):
            print(f"{i + 1}. {option}")
            
        while True:
            try:
                # Get user input and validate range
                choice = int(input("Enter number: "))
                if 1 <= choice <= len(options):
                    return choice - 1  # Return 0-based index
                else:
                    print(f"Please enter a number between 1 and {len(options)}.")
            except ValueError:
                print("Please enter a valid number.")
                
            