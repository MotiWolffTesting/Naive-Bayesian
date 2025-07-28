# Naive Bayes Classifier Project (v2)

## Overview

This project implements a Naive Bayes classifier from scratch in Python, designed to classify tabular data (such as CSV files) using supervised learning. The project is built with a modular architecture and deployed using two separate Docker containers - one for the classifier client and another for the model management API.

## Features

- **Naive Bayes algorithm** with Laplace smoothing
- **Modular architecture**: Classifier and Model Management components
- **Two-container deployment**: Separate containers for client and API
- **Multiple interfaces**: Console client, Streamlit web interface, and FastAPI REST API
- **Automated workflows**: Non-interactive training and testing
- **Caching system**: Persistent JSON cache for model results
- **Validation utilities**: Train-test splits and confusion matrix computation
- **Docker support** for easy deployment
- Train on one dataset, test on another (supports standard train/test splits)
- Classify individual records or entire datasets
- Comprehensive error handling and input validation
- Support for large CSV files (up to 100MB)

## Project Structure

```
Naive-Baysian/
├── classifier/
│   ├── engine.py                   # Main classification orchestration
│   └── classifier.py               # Naive Bayes classifier implementation
├── model_management/
│   ├── builder.py                  # Model training logic
│   ├── cleaner.py                  # Data cleaning utilities
│   ├── data_loader.py              # Data loading utilities
│   ├── model.py                    # Model parameter storage
│   └── validator.py                # Validation utilities
├── api/
│   └── api_server.py              # FastAPI server
├── UI/
│   ├── console_api_client.py      # Console API client
│   └── streamlit_client.py        # Streamlit web interface
├── data/                          # Dataset files
├── main.py                        # Automated console application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # API server Docker configuration
├── README.md
└── results_cache.json             # Cached model results
```

## Architecture

### v2: Split Container Architecture

The project uses two separate Docker containers:

1. **Model Management API Container** (`model-api`):
   - FastAPI server providing REST endpoints
   - Handles model training, testing, and prediction
   - Includes caching system for results
   - Runs on port 8000

2. **Classifier Client Container** (`classifier-client`):
   - Automated client that runs `main.py`
   - Communicates with API container via HTTP
   - Stays alive after execution for debugging
   - Uses hardcoded dataset paths

## Getting Started

### Prerequisites

- Docker (for containerized deployment)
- Python 3.7+ (for local development)

### Option 1: Docker Deployment (Recommended)

1. **Build the API server image:**
   ```bash
   docker build -t model-api .
   ```

2. **Build the client image:**
   ```bash
   docker build -f Dockerfile.client -t classifier-client .
   ```

3. **Create a Docker network:**
   ```bash
   docker network create naive-bayes-network
   ```

4. **Run the API server container:**
   ```bash
   docker run -d --name model-api --network naive-bayes-network -p 8000:8000 model-api
   ```

5. **Run the client container:**
   ```bash
   docker run -d --name classifier-client --network naive-bayes-network classifier-client
   ```

6. **Check the logs:**
   ```bash
   docker logs classifier-client
   ```

### Option 2: Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API server:**
   ```bash
   uvicorn api.api_server:app --reload
   ```

3. **Run the automated client:**
   ```bash
   python main.py
   ```

4. **Or run the Streamlit interface:**
   ```bash
   streamlit run UI/streamlit_client.py
   ```

## Usage

### Automated Console Client (`main.py`)

The console application runs automatically with hardcoded settings:
- **Training dataset**: `data/mushroom_train.csv`
- **Target column**: `edible`
- **Workflow**: Train model → Test accuracy → Display results
- **Caching**: Results are cached in `results_cache.json`

### Streamlit Web Interface

The Streamlit interface provides an automated workflow:
1. **Automatic training** with hardcoded dataset
2. **Automatic testing** and accuracy display
3. **Model information** display
4. **Sample classification** demonstration

### FastAPI Server

The API server runs on `http://localhost:8000` and provides the following endpoints:

#### POST `/train`
Train the model with a CSV file.
- **Parameters**: 
  - `file`: CSV file upload
  - `target_column`: Name of the target column (form data)
- **Response**: Training status and cache information

#### POST `/predict`
Classify a single record.
- **Body**: JSON object with feature values
- **Response**: Predicted class and confidence

#### POST `/test`
Test model accuracy with a CSV file.
- **Parameters**:
  - `file`: CSV file upload
  - `target_column`: Name of the target column (optional, form data)
- **Response**: Accuracy, confusion matrix, and cache status

#### GET `/info`
Get model information and statistics.

### Example API Usage

```bash
# Train the model
curl -X POST "http://localhost:8000/train" \
  -F "file=@data/mushroom_train.csv" \
  -F "target_column=edible"

# Predict a record
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"cap-shape": "b", "cap-surface": "s", "cap-color": "n"}'

# Test accuracy
curl -X POST "http://localhost:8000/test" \
  -F "file=@data/mushroom_test.csv" \
  -F "target_column=edible"

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
cap-shape,cap-surface,cap-color,edible
b,s,n,e
c,s,y,p
```

## Configuration

### Environment Variables
- `MAX_FILE_SIZE`: Maximum file size for uploads (default: 100MB)
- `SUPPORTED_FORMATS`: Supported file formats (default: ['.csv'])
- `API_URL`: API server URL (default: http://127.0.0.1:8000)

### Docker Configuration
- **API Server Port**: 8000
- **Network**: `naive-bayes-network` for inter-container communication
- **Client Container**: Stays alive after execution for debugging

## Caching System

The project includes a robust caching system:
- **Cache file**: `results_cache.json`
- **Cache key**: SHA256 hash of file content + target column
- **Cached data**: Training status, accuracy, confusion matrix
- **Benefits**: Avoids redundant training for same datasets

## Development

### Code Structure
- **Classifier Module**: Core classification logic and orchestration
- **Model Management Module**: Training, cleaning, validation, and data loading
- **API Server**: FastAPI implementation with caching
- **UI Components**: Console and Streamlit interfaces

### Key Components
- **ClassificationEngine**: Orchestrates model building and classification
- **NaiveBayesClassifier**: Core algorithm implementation
- **Cleaner**: Data cleaning utilities (Laplace smoothing)
- **Validator**: Train-test splits and confusion matrix computation
- **Builder**: Model training logic

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
- Network connectivity issues

## Troubleshooting

### Common Issues

1. **Container Communication**: Ensure both containers are on the same Docker network
2. **Port Conflicts**: Check if port 8000 is available
3. **Cache Issues**: Delete `results_cache.json` to reset cache
4. **Missing Dependencies**: Rebuild Docker images if needed

### Debugging

- **API Server Logs**: `docker logs model-api`
- **Client Logs**: `docker logs classifier-client`
- **Network Issues**: `docker network inspect naive-bayes-network`

## License

This project is for educational purposes.

---

For questions or suggestions, please open an issue or contact the maintainers.
