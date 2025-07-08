from abc import abstractmethod, ABC
from typing import List

class UserInterface(ABC):
    """Abstract user interface"""
    
    @abstractmethod
    def display_message(self, message: str):
        pass
    
    @abstractmethod
    def get_user_input(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def get_menu_choice(self, options: List[str]) -> int:
        pass
        
    