# FastAPI server for Naive Bayes classifier API
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from classifier.engine import ClassificationEngine
from model_management.data_loader import DataLoader
import hashlib
import json
import os
from model_management.validator import Validator

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB file size limit
SUPPORTED_FORMATS = ['.csv']
CACHE_FILE = 'results_cache.json'  # Cache file for results

app = FastAPI() # Create FastAPI app
engine = ClassificationEngine() # In-memory model engine

# Compute a unique hash for a file and target column
def get_file_hash(file_bytes, target_column):
    hasher = hashlib.sha256()
    hasher.update(file_bytes)
    hasher.update(target_column.encode('utf-8'))
    return hasher.hexdigest()

# Load results cache from disk
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save results cache to disk
def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

# Read and validate uploaded CSV file
def read_csv_upload(upload_file: UploadFile) -> pd.DataFrame:
    if not upload_file.filename.lower().endswith('.csv'):
        raise ValueError("File must be a CSV file")
    if hasattr(upload_file, 'size') and upload_file.size and upload_file.size > MAX_FILE_SIZE:
        raise ValueError("File size exceeds 100MB limit")
    try:
        df = pd.read_csv(upload_file.file)
        if df.empty:
            raise ValueError("CSV file is empty")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")

# Train endpoint: builds model and caches results
@app.post("/train")
async def train(file: UploadFile = File(...), target_column: str = Form(...)):
    try:
        file_bytes = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
        file_hash = get_file_hash(file_bytes, target_column)
        cache = load_cache()
        # Always (re)train the in-memory model, even if cached
        engine.build_model(df, target_column)
        if file_hash in cache:
            # Return cached status if model already built
            return {"status": "Model trained (cached)", "target_column": target_column, "cached": True}
        # If not cached, cache it
        cache[file_hash] = {"status": "trained", "target_column": target_column}
        save_cache(cache)
        return {"status": "Model trained", "target_column": target_column, "cached": False}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

# Predict endpoint: classify a single record
@app.post("/predict")
async def predict(record: dict):
    try:
        if not engine.is_model_ready():
            return JSONResponse(status_code=400, content={"error": "Model is not trained yet"})
        if not record or not isinstance(record, dict):
            return JSONResponse(status_code=400, content={"error": "Record must be a non-empty dictionary"})
        result = engine.classify_single_record(record=record)
        return {"prediction": result}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

# Test endpoint: evaluate model accuracy and confusion matrix, with caching
@app.post("/test")
async def test(file: UploadFile = File(...), target_column: str = Form(None)):
    try:
        file_bytes = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
        if not engine.is_model_ready():
            return JSONResponse(status_code=400, content={"error": "Model is not trained yet"})
        if target_column and target_column.strip():
            if target_column not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found in test data"})
        file_hash = get_file_hash(file_bytes, target_column or "")
        cache = load_cache()
        # Return cached results if available
        if file_hash in cache and "accuracy" in cache[file_hash] and "confusion_matrix" in cache[file_hash]:
            return {"accuracy": cache[file_hash]["accuracy"], "confusion_matrix": cache[file_hash]["confusion_matrix"], "cached": True}
        # Compute accuracy and confusion matrix
        x_test = df.drop(columns=[target_column])
        y_test = df[target_column]
        predictions = engine._classifier.classify_group(x_test)
        validator = Validator()
        cm = validator.compute_confusion_matrix(y_test, predictions).tolist()
        accuracy = sum(1 for prediction, actual in zip(predictions, y_test) if prediction == actual) / len(y_test)
        cache[file_hash] = cache.get(file_hash, {}) 
        cache[file_hash]["accuracy"] = accuracy
        cache[file_hash]["confusion_matrix"] = cm
        save_cache(cache)
        return {"accuracy": accuracy, "confusion_matrix": cm, "cached": False}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

# Info endpoint: return model metadata
@app.get("/info")
async def info():
    import numpy as np
    info = engine.get_classifier_info()
    # Recursively convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    return JSONResponse(content=convert(info))