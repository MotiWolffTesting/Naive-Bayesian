from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from Classification.classification_engine import ClassificationEngine
from utils.data_loader import DataLoader

app = FastAPI() # Initiating FastAPI App
engine = ClassificationEngine() # Initiating Engine

def read_csv_upload(upload_file: UploadFile) -> pd.DataFrame:
    return pd.read_csv(upload_file.file)

@app.post("/train")
async def train(file: UploadFile = File(...), target_column: str = Form(...)):
    """Train the model with a csv file and a target column"""
    df = read_csv_upload(file)
    if target_column not in df.columns:
        return JSONResponse(status_code=400, content={"error": "Invalid target column"})
    engine.build_model(df, target_column)
    return {"status": "Model trained", "target column": target_column}

@app.post("/predict")
async def predict(record: dict):
    """Classify a single record (JSON)"""
    if not engine.is_model_ready:
        return JSONResponse(status_code=400, content={"error": "Model is not trained yet."})
    try:
        result = engine.classify_single_record(record=record)
        return {"prediction": result}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
@app.post("/test")
async def test(file: UploadFile = File(...)):
    """Test model accuracy with csv file"""
    if not engine.is_model_ready:
        return JSONResponse(status_code=400, content={"error": "Model is not trained yet."})
    df = read_csv_upload(file)
    try:
        accuracy = engine.test_model_accuracy(df)
        return {"accuracy": accuracy}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    
@app.get("/info")
async def info():
    """Get model info"""
    return engine.get_classifier_info()