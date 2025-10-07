"""
FastAPI REST API for Bank Marketing Term Deposit Prediction
Provides programmatic access to the prediction model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import Optional, Dict
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bank Marketing Term Deposit Prediction API",
    description="REST API for predicting term deposit subscription",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths
MODEL_PATH = "../models/lightgbm_retrained_tuned.pkl"
SCALER_PATH = "../models/preprocessing/scaler.pkl"
ENCODERS_PATH = "../models/preprocessing/label_encoders.pkl"

# Global variables for model
model = None
scaler = None
encoders = None

class ClientData(BaseModel):
    """Input data schema for prediction"""
    age: int = Field(..., ge=18, le=95, description="Client age")
    job: str = Field(..., description="Job type")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default?")
    housing: str = Field(..., description="Has housing loan?")
    loan: str = Field(..., description="Has personal loan?")
    contact: str = Field(..., description="Contact communication type")
    month: str = Field(..., description="Last contact month")
    day_of_week: str = Field(..., description="Last contact day of week")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Number of contacts in this campaign")
    pdays: int = Field(..., ge=0, description="Days since last contact (999=not contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts in previous campaigns")
    poutcome: str = Field(..., description="Outcome of previous campaign")
    emp_var_rate: float = Field(..., description="Employment variation rate")
    cons_price_idx: float = Field(..., description="Consumer price index")
    cons_conf_idx: float = Field(..., description="Consumer confidence index")
    euribor3m: float = Field(..., description="Euribor 3 month rate")
    nr_employed: float = Field(..., description="Number of employees")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "job": "admin.",
                "marital": "single",
                "education": "university.degree",
                "default": "no",
                "housing": "no",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 180,
                "campaign": 1,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.1,
                "cons_price_idx": 93.994,
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191.0
            }
        }

class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    prediction: str = Field(..., description="Prediction result (yes/no)")
    probability: float = Field(..., description="Probability of subscription")
    confidence: str = Field(..., description="Confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str

@app.on_event("startup")
async def load_model():
    """Load model and preprocessors on startup"""
    global model, scaler, encoders
    try:
        logger.info("Loading model and preprocessors...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Bank Marketing Term Deposit Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: ClientData):
    """
    Predict term deposit subscription
    
    - **age**: Client age (18-95)
    - **job**: Job type
    - **marital**: Marital status
    - **education**: Education level
    - **default**: Has credit in default
    - **housing**: Has housing loan
    - **loan**: Has personal loan
    - **contact**: Contact type
    - **month**: Last contact month
    - **day_of_week**: Last contact day of week
    - **duration**: Last contact duration in seconds
    - **campaign**: Number of contacts in this campaign
    - **pdays**: Days since last contact (999=not contacted)
    - **previous**: Number of contacts in previous campaigns
    - **poutcome**: Outcome of previous campaign
    - **emp_var_rate**: Employment variation rate
    - **cons_price_idx**: Consumer price index
    - **cons_conf_idx**: Consumer confidence index
    - **euribor3m**: Euribor 3 month rate
    - **nr_employed**: Number of employees
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create input dataframe
        input_data = pd.DataFrame({
            'age': [data.age],
            'job': [data.job],
            'marital': [data.marital],
            'education': [data.education],
            'default': [data.default],
            'housing': [data.housing],
            'loan': [data.loan],
            'contact': [data.contact],
            'month': [data.month],
            'day_of_week': [data.day_of_week],
            'duration': [data.duration],
            'campaign': [data.campaign],
            'pdays': [data.pdays],
            'previous': [data.previous],
            'poutcome': [data.poutcome],
            'emp.var.rate': [data.emp_var_rate],
            'cons.price.idx': [data.cons_price_idx],
            'cons.conf.idx': [data.cons_conf_idx],
            'euribor3m': [data.euribor3m],
            'nr.employed': [data.nr_employed]
        })
        
        # Encode categorical features
        categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                           'contact', 'month', 'day_of_week', 'poutcome']
        
        for col in categorical_cols:
            if col in encoders:
                # Handle unknown categories
                if input_data[col].iloc[0] not in encoders[col].classes_:
                    logger.warning(f"Unknown category '{input_data[col].iloc[0]}' for {col}, using default")
                    input_data[col] = encoders[col].classes_[0]
                input_data[col] = encoders[col].transform(input_data[col])
        
        # Scale numerical features
        numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                         'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                         'euribor3m', 'nr.employed']
        
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Interpret results
        result = "yes" if prediction == 1 else "no"
        prob_yes = float(probability[1])
        
        if prob_yes >= 0.7:
            confidence = "high"
        elif prob_yes >= 0.5:
            confidence = "moderate"
        else:
            confidence = "low"
        
        logger.info(f"Prediction: {result}, Probability: {prob_yes:.4f}")
        
        return PredictionResponse(
            prediction=result,
            probability=prob_yes,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "LightGBM",
        "version": "1.0.0",
        "features": {
            "categorical": ["job", "marital", "education", "default", "housing", 
                          "loan", "contact", "month", "day_of_week", "poutcome"],
            "numerical": ["age", "duration", "campaign", "pdays", "previous", 
                        "emp.var.rate", "cons.price.idx", "cons.conf.idx", 
                        "euribor3m", "nr.employed"]
        },
        "target": "term deposit subscription (yes/no)",
        "dataset": "UCI Bank Marketing Dataset",
        "samples": "86,000+",
        "performance": {
            "metric": "ROC-AUC",
            "value": "~0.93"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
