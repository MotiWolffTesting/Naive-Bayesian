from UI.console_api_client import train_model, test_model

# Hardcoded training data path and target column
TRAINING_CSV_PATH = 'data/tennis_train.csv'
TARGET_COLUMN = 'play_tennis'

def main():
    # Train the model (non-interactive, uses API)
    train_model(TRAINING_CSV_PATH, TARGET_COLUMN)
    # Test the model (non-interactive, uses API)
    test_model(TRAINING_CSV_PATH, TARGET_COLUMN)

if __name__ == "__main__":
    main()