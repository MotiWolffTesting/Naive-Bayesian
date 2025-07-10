from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from Classification.classification_engine import ClassificationEngine
from utils.data_loader import DataLoader

# Constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_FORMATS = ['.csv']

app = FastAPI() # Initiating FastAPI App
engine = ClassificationEngine() # Initiating Engine

def read_csv_upload(upload_file: UploadFile) -> pd.DataFrame:
    """Read and validate CSV upload"""
    # Validate file type
    if not upload_file.filename.lower().endswith('.csv'):
        raise ValueError("File must be a CSV file")
    
    # Validate file size
    if upload_file.size and upload_file.size > MAX_FILE_SIZE:
        raise ValueError("File size exceeds 10MB limit")
    
    try:
        df = pd.read_csv(upload_file.file)
        if df.empty:
            raise ValueError("CSV file is empty")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {e}")

@app.post("/train")
async def train(file: UploadFile = File(...), target_column: str = Form(...)):
    """Train the model with a csv file and a target column"""
    try:
        df = read_csv_upload(file)
        
        # Validate target column
        if not target_column or not target_column.strip():
            return JSONResponse(status_code=400, content={"error": "Target column cannot be empty"})
        
        if target_column not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found in data"})
        
        # Build model
        if engine.build_model(df, target_column):
            return {"status": "Model trained", "target_column": target_column}
        else:
            return JSONResponse(status_code=500, content={"error": "Failed to train model"})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

@app.post("/predict")
async def predict(record: dict):
    """Classify a single record (JSON)"""
    try:
        # Validate model is ready
        if not engine.is_model_ready():
            return JSONResponse(status_code=400, content={"error": "Model is not trained yet"})
        
        # Validate input record
        if not record or not isinstance(record, dict):
            return JSONResponse(status_code=400, content={"error": "Record must be a non-empty dictionary"})
        
        result = engine.classify_single_record(record=record)
        return {"prediction": result}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})
    
@app.post("/test")
async def test(file: UploadFile = File(...), target_column: str = Form(None)):
    """
    Test model accuracy with a CSV file.
    """
    try:
        # Validate model is ready
        if not engine.is_model_ready():
            return JSONResponse(status_code=400, content={"error": "Model is not trained yet"})
        
        # Read and validate test data
        df = read_csv_upload(file)
        
        # Validate target column if provided
        if target_column and target_column.strip():
            if target_column not in df.columns:
                return JSONResponse(status_code=400, content={"error": f"Target column '{target_column}' not found in test data"})
        
        accuracy = engine.test_model_accuracy(df, target_column=target_column)
        return {"accuracy": accuracy}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})
    
@app.get("/info")
async def info():
    """Get model info"""
    return engine.get_classifier_info()