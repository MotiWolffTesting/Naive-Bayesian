# Naive Bayes Classifier Project

## Overview

This project implements a Naive Bayes classifier for categorical/tabular data, featuring:
- A FastAPI server for all model/data operations (model-api container)
- A client container (classifier-client) that interacts with the API (can be automated or interactive)
- Modular, maintainable codebase with clear separation of concerns
- Dockerized deployment for both API and client
- Caching of results to avoid redundant computation

---

## Project Structure

```
Naive-Baysian/
├── api/
│   └── api_server.py              # FastAPI server exposing model endpoints
├── classifier/
│   ├── engine.py                  # Orchestrates model operations (API-side)
│   └── classifier.py              # Core Naive Bayes algorithm
├── model_management/
│   ├── builder.py                 # Model builder/trainer
│   ├── cleaner.py                 # Data cleaning (Laplace, etc.)
│   ├── data_loader.py             # CSV data loading utilities
│   ├── model.py                   # Model data structure
│   └── validator.py               # Validation (split, confusion matrix)
├── UI/
│   └── console_api_client.py      # CLI client for API interaction (automated or interactive)
├── data/                          # Example datasets (CSV)
├── main.py                        # Automated API client entry point
├── Dockerfile                     # Docker build for API
├── Dockerfile.client              # Docker build for client
├── default_cmd.sh                 # Entrypoint script for client container
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## How It Works (v2: Split Containers)

- **model-api container**: Runs the FastAPI server. Handles all data loading, model building, validation, and prediction.
- **classifier-client container**: Runs the client code (main.py, UI/console_api_client.py). Sends requests to the API server for all operations.
- **Communication**: The client container connects to the API container using the Docker network (API_URL is set to `http://model-api:8000`).
- **Caching**: The API server caches results for each unique dataset/target column combination.

---

## Building and Running (Without docker-compose)

### 1. Build the API Server Image
```sh
docker build -t model-api -f Dockerfile .
```

### 2. Build the Client Image
```sh
docker build -t classifier-client -f Dockerfile.client .
```

### 3. Create a Docker Network
```sh
docker network create nb-network
```

### 4. Run the API Server Container
```sh
docker run -d --name model-api --network nb-network -p 8000:8000 model-api
```

### 5. Run the Client Container
```sh
docker run --name classifier-client --network nb-network -e API_URL=http://model-api:8000 classifier-client
```
- The client will run `main.py` and then stay alive for inspection.
- Both containers can access the `data/` directory if you add `-v $(pwd)/data:/app/data` to both run commands.

---

## API Endpoints

- **POST `/train`**: Train the model with a CSV file and target column (caching enabled)
- **POST `/predict`**: Classify a single record (JSON)
- **POST `/test`**: Test model accuracy with a CSV file (caching enabled)
- **GET `/info`**: Get model information and statistics

---

## Example API Usage (from host or client container)

```sh
# Train the model
curl -X POST "http://localhost:8000/train" \
  -F "file=@data/mushroom_train.csv" \
  -F "target_column=edible"

# Test accuracy
curl -X POST "http://localhost:8000/test" \
  -F "file=@data/mushroom_train.csv" \
  -F "target_column=edible"

# Predict a record
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"cap_shape": "x", "cap_surface": "s", ...}'

# Get model info
curl "http://localhost:8000/info"
```

---

## Data Format

- CSV files with features as columns and a target column for class labels.
- Example:
  ```csv
  cap_shape,cap_surface,cap_color,bruises,odor,edible
  x,s,n,t,p,e
  x,y,w,t,p,p
  ...
  ```

---

## Configuration

- **API_URL**: Set in the client container to `http://model-api:8000` (via environment variable)
- **Data directory**: Mount `./data` to `/app/data` in both containers for shared access

---

## Development & Extensibility

- Modular codebase: Easy to add new classifiers, UIs, or API endpoints
- Caching: Results are cached by dataset and target column
- Error handling: All endpoints validate input and provide clear error messages
- Docker: Both API and client are fully containerized

---

## License

This project is for educational purposes.

---

For questions or suggestions, please open an issue or contact the maintainers.
