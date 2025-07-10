# Naive Bayes Classifier Project

## Overview

This project implements a Naive Bayes classifier from scratch in Python, designed to classify tabular data (such as CSV files) using supervised learning. The project provides multiple interfaces including a console application, FastAPI server, and Docker containerization for easy deployment and usage.

## Features

- **Naive Bayes algorithm** with Laplace smoothing
- **Multiple interfaces**: Console UI and FastAPI REST API
- **Docker support** for easy deployment
- Train on one dataset, test on another (supports standard train/test splits)
- Classify individual records or entire datasets
- Modular, object-oriented codebase with clear separation of concerns
- Comprehensive error handling and input validation
- Support for large CSV files (up to 100MB)

## Project Structure

```
Naive-Baysian/
├── Classification/
│   ├── classification_engine.py    # Main classification logic
│   └── naive_bayes_classifier.py   # Naive Bayes implementation
├── api/
│   └── api_server.py              # FastAPI server
├── UI/
│   ├── console_interface.py       # Console UI
│   └── user_interface.py          # API client interface
├── utils/
│   └── data_loader.py             # Data loading utilities
├── main.py                        # Console application entry point
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.7+ (for local development)
- Docker (for containerized deployment)

### Option 1: Local Development

1. **Clone or download the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the console application:**
   ```bash
   python main.py
   ```

4. **Or run the FastAPI server:**
   ```bash
   uvicorn api.api_server:app --reload
   ```

### Option 2: Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Or build and run manually:**
   ```bash
   docker build -t naive-bayes-classifier .
   docker run -p 8000:8000 naive-bayes-classifier
   ```

## Usage

### Console Interface

The console application provides an interactive menu:

1. **Train Model**: Upload a CSV file and specify the target column
2. **Test Accuracy**: Test the model with a separate CSV file
3. **Classify Record**: Predict the class of a single record
4. **View Model Info**: Display model statistics and parameters

### FastAPI Server

The API server runs on `http://localhost:8000` and provides the following endpoints:

#### POST `/train`
Train the model with a CSV file.
- **Parameters**: 
  - `file`: CSV file upload
  - `target_column`: Name of the target column (form data)

#### POST `/predict`
Classify a single record.
- **Body**: JSON object with feature values

#### POST `/test`
Test model accuracy with a CSV file.
- **Parameters**:
  - `file`: CSV file upload
  - `target_column`: Name of the target column (optional, form data)

#### GET `/info`
Get model information and statistics.

### Example API Usage

```bash
# Train the model
curl -X POST "http://localhost:8000/train" \
  -F "file=@phishing.csv" \
  -F "target_column=class"

# Predict a record
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"feature1": "value1", "feature2": "value2"}'

# Test accuracy
curl -X POST "http://localhost:8000/test" \
  -F "file=@test_data.csv" \
  -F "target_column=class"

# Get model info
curl "http://localhost:8000/info"
```

## Data Format

The classifier expects CSV files with:
- **Features**: Any number of columns containing feature values
- **Target Column**: One column containing the class labels
- **Data Types**: Supports both categorical and numerical features

### Example CSV Structure
```csv
feature1,feature2,feature3,class
value1,value2,value3,positive
value4,value5,value6,negative
```

## Configuration

### Environment Variables
- `MAX_FILE_SIZE`: Maximum file size for uploads (default: 100MB)
- `SUPPORTED_FORMATS`: Supported file formats (default: ['.csv'])

### Docker Configuration
- **Port**: 8000 (configurable in docker-compose.yml)
- **Memory**: Optimized for containerized deployment
- **Dependencies**: All requirements included in container

## Development

### Code Structure
- **Classification Engine**: Main orchestration logic
- **Naive Bayes Classifier**: Core algorithm implementation
- **Data Loader**: CSV parsing and validation
- **UI Components**: Console and API interfaces
- **API Server**: FastAPI implementation with comprehensive error handling

### Adding New Features
The modular design makes it easy to:
- Add new classification algorithms
- Implement additional UI interfaces
- Extend API endpoints
- Add new data preprocessing steps

## Error Handling

The application includes comprehensive error handling for:
- Invalid file formats
- Missing target columns
- Empty datasets
- Model training failures
- API request validation
- File size limits

## License

This project is for educational purposes.

---

For questions or suggestions, please open an issue or contact the maintainers.
