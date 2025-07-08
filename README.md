# Naive Bayes Classifier Project

## Overview

This project implements a Naive Bayes classifier from scratch in Python, designed to classify tabular data (such as CSV files) using supervised learning. The main goal is to provide a simple, object-oriented framework for experimenting with classification tasks, with a focus on clear code structure and usability.

## Features

- **Naive Bayes algorithm** with Laplace smoothing
- Train on one dataset, test on another (supports standard train/test splits)
- Classify individual records or entire datasets
- Console-based user interface (easy to extend or swap for a GUI)
- Modular, object-oriented codebase

## Getting Started

### Prerequisites

- Python 3.7+
- `pandas` and `numpy` libraries

You can install the required packages with:

```bash
pip install pandas numpy
```

### Running the Program

1. **Clone or download the repository.**
2. **Navigate to the project directory.**
3. **Run the main script:**

   ```bash
   python main.py
   ```

4. **Follow the on-screen prompts:**
   - Enter the path to your training CSV file.
   - Select the target column (the label you want to predict).
   - After the model is built, you can:
     - Test accuracy with a separate test CSV file.
     - Classify a single new record.
     - View model information.

### Example

Suppose you have `mushroom_train.csv` and `mushroom_test.csv` in a `tests/` folder:

- When prompted for the training file, enter:  
  `tests/mushroom_train.csv`
- For the target column, enter:  
  `edible`
- To test accuracy, enter:  
  `tests/mushroom_test.csv`

### Project Structure

```
Naive-Baysian/
  ├── Classification/
  │   ├── classification_engine.py
  │   └── naive_bayes_classifier.py
  ├── UI/
  │   ├── console_interface.py
  │   └── user_interface.py
  ├── utils/
  │   └── data_loader.py
  └── main.py
```

### Notes

- The program expects the target column name (not its index).
- You can split your data into train/test sets using Excel or any tool, or automate it in code if you wish.
- The code is designed to be easy to extend (e.g., adding a GUI or new classifiers).

## License

This project is for educational purposes.

---

If you have any questions or suggestions, feel free to open an issue or contact the author.
