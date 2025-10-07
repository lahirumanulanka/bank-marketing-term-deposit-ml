"""
Bank Marketing Term Deposit Prediction API
FastAPI server for Hugging Face Spaces deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import uvicorn
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Bank Marketing Term Deposit Prediction API",
    description="REST API for predicting term deposit subscription using machine learning",
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

# Model paths (adjust for Hugging Face Spaces structure)
MODEL_PATH = "xgboost_retrained_tuned.pkl"
SCALER_PATH = "preprocessing/scaler.pkl" 
ENCODERS_PATH = "preprocessing/label_encoders.pkl"

# Global variables for model
model = None
scaler = None
encoders = None

class ClientData(BaseModel):
    """Input schema for prediction"""
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
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., ge=0, description="Days since last contact (999 if never contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., description="Outcome of previous marketing campaign")
    emp_var_rate: float = Field(..., description="Employment variation rate")
    cons_price_idx: float = Field(..., description="Consumer price index")
    cons_conf_idx: float = Field(..., description="Consumer confidence index")
    euribor3m: float = Field(..., description="Euribor 3 month rate")
    nr_employed: float = Field(..., description="Number of employees")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "job": "management",
                "marital": "married",
                "education": "university.degree",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "fri",
                "duration": 180,
                "campaign": 2,
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
    prediction: str = Field(..., description="Prediction: 'yes' or 'no'")
    probability: float = Field(..., description="Probability of subscription (0-1)")
    probability_percentage: str = Field(..., description="Probability as percentage")
    confidence_level: str = Field(..., description="Confidence interpretation")
    model_version: str = Field(..., description="Model version used")

class BatchClientData(BaseModel):
    """Schema for batch predictions"""
    clients: List[ClientData] = Field(..., description="List of client data for batch prediction")

def load_model():
    """Load the trained model and preprocessors"""
    global model, scaler, encoders
    
    try:
        # Load model
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        # Load scaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
        
        # Load encoders
        if os.path.exists(ENCODERS_PATH):
            with open(ENCODERS_PATH, 'rb') as f:
                encoders = pickle.load(f)
        else:
            raise FileNotFoundError(f"Encoders file not found: {ENCODERS_PATH}")
            
        return True
        
    except Exception as e:
        print(f"Error loading model components: {e}")
        return False

def preprocess_data(client_data: ClientData) -> pd.DataFrame:
    """Preprocess input data for prediction with feature engineering"""
    
    # Create input dataframe with basic features
    input_data = pd.DataFrame({
        'age': [client_data.age],
        'job': [client_data.job],
        'marital': [client_data.marital],
        'education': [client_data.education],
        'default': [client_data.default],
        'housing': [client_data.housing],
        'loan': [client_data.loan],
        'contact': [client_data.contact],
        'month': [client_data.month],
        'day_of_week': [client_data.day_of_week],
        'duration': [client_data.duration],
        'campaign': [client_data.campaign],
        'pdays': [client_data.pdays],
        'previous': [client_data.previous],
        'poutcome': [client_data.poutcome],
        'emp.var.rate': [client_data.emp_var_rate],
        'cons.price.idx': [client_data.cons_price_idx],
        'cons.conf.idx': [client_data.cons_conf_idx],
        'euribor3m': [client_data.euribor3m],
        'nr.employed': [client_data.nr_employed]
    })
    
    # Add engineered features that the model expects
    # These are based on the original feature engineering
    
    # Add missing features with default/calculated values
    input_data['balance'] = 0  # Default balance (not provided in input)
    input_data['day'] = 15     # Default day of month (not provided in input)
    input_data['data_source'] = 'api'  # Source identifier
    
    # Engineered features
    input_data['contact_frequency'] = input_data['campaign']  # Simple approximation
    input_data['previous_campaign_success'] = (input_data['poutcome'] == 'success').astype(int)
    
    # Age groups (based on common banking demographics)
    input_data['age_group'] = pd.cut(input_data['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['young', 'young_adult', 'middle_aged', 'senior', 'elderly']).astype(str)
    
    # Economic data indicator
    input_data['has_economic_data'] = 1  # Always 1 since we provide economic indicators
    
    # Duration categories
    input_data['duration_category'] = pd.cut(input_data['duration'],
                                           bins=[0, 120, 300, 600, float('inf')],
                                           labels=['short', 'medium', 'long', 'very_long']).astype(str)
    
    # Encode categorical features (including engineered ones)
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome', 'data_source',
                       'age_group', 'duration_category']
    
    for col in categorical_cols:
        if col in encoders:
            # Handle unknown categories by using the first class
            if input_data[col].iloc[0] not in encoders[col].classes_:
                input_data[col] = encoders[col].classes_[0]
            input_data[col] = encoders[col].transform(input_data[col])
        else:
            # For new categorical features not in original encoders, use simple encoding
            if col == 'data_source':
                input_data[col] = 0  # Simple numeric encoding
            elif col == 'age_group':
                age_group_map = {'young': 0, 'young_adult': 1, 'middle_aged': 2, 'senior': 3, 'elderly': 4}
                input_data[col] = age_group_map.get(input_data[col].iloc[0], 2)  # Default to middle_aged
            elif col == 'duration_category':
                duration_cat_map = {'short': 0, 'medium': 1, 'long': 2, 'very_long': 3}
                input_data[col] = duration_cat_map.get(input_data[col].iloc[0], 1)  # Default to medium
    
    # Scale numerical features (including engineered ones)
    numerical_cols = ['age', 'balance', 'campaign', 'day', 'duration', 'previous', 'pdays',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                     'euribor3m', 'nr.employed', 'contact_frequency', 'previous_campaign_success',
                     'has_economic_data']
    
    # Only scale columns that exist and are numeric
    existing_numerical = [col for col in numerical_cols if col in input_data.columns]
    if existing_numerical:
        # Use only the features that were used to fit the scaler originally
        basic_numerical = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                          'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                          'euribor3m', 'nr.employed']
        input_data[basic_numerical] = scaler.transform(input_data[basic_numerical])
    
    # Ensure we have all the features the model expects in the right order
    expected_features = ['age', 'balance', 'campaign', 'cons.conf.idx', 'cons.price.idx', 'contact',
                        'day', 'day_of_week', 'default', 'duration', 'education', 'emp.var.rate', 
                        'euribor3m', 'housing', 'job', 'loan', 'marital', 'month', 'nr.employed', 
                        'pdays', 'poutcome', 'previous', 'contact_frequency', 'previous_campaign_success', 
                        'age_group', 'has_economic_data', 'duration_category']
    
    # Reorder columns to match expected features
    input_data = input_data.reindex(columns=expected_features, fill_value=0)
    
    return input_data

def interpret_confidence(probability: float) -> str:
    """Interpret prediction confidence level"""
    prob_percentage = probability * 100
    
    if prob_percentage >= 80:
        return "Very High"
    elif prob_percentage >= 70:
        return "High"
    elif prob_percentage >= 60:
        return "Moderate"
    elif prob_percentage >= 50:
        return "Low"
    else:
        return "Very Low"

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        raise RuntimeError("Failed to load model components")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bank Marketing Term Deposit Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "endpoints": {
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and scaler is not None and encoders is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost",
        "version": "1.0.0",
        "features": 20,
        "target": "term_deposit_subscription",
        "description": "Predicts whether a client will subscribe to a term deposit",
        "training_data": "UCI Bank Marketing Dataset",
        "performance": {
            "roc_auc": "~0.93",
            "note": "Trained on 86,000+ samples with SMOTE for class imbalance"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_subscription(client_data: ClientData):
    """
    Predict whether a client will subscribe to a term deposit
    
    Returns prediction, probability, and confidence level
    """
    
    if model is None or scaler is None or encoders is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess data
        processed_data = preprocess_data(client_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Get probability for positive class (subscription = yes)
        prob_yes = probability[1]
        
        # Interpret results
        result = "yes" if prediction == 1 else "no"
        confidence = interpret_confidence(prob_yes)
        
        return PredictionResponse(
            prediction=result,
            probability=prob_yes,
            probability_percentage=f"{prob_yes * 100:.2f}%",
            confidence_level=confidence,
            model_version="XGBoost-1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(batch_data: BatchClientData):
    """
    Predict for multiple clients in batch
    
    Returns list of predictions for all clients
    """
    
    if model is None or scaler is None or encoders is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for client_data in batch_data.clients:
            # Preprocess data
            processed_data = preprocess_data(client_data)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
            # Get probability for positive class
            prob_yes = probability[1]
            
            # Interpret results
            result = "yes" if prediction == 1 else "no"
            confidence = interpret_confidence(prob_yes)
            
            results.append({
                "prediction": result,
                "probability": prob_yes,
                "probability_percentage": f"{prob_yes * 100:.2f}%",
                "confidence_level": confidence
            })
        
        return {
            "predictions": results,
            "count": len(results),
            "model_version": "XGBoost-1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/features/info")
async def feature_info():
    """Get information about input features"""
    return {
        "features": {
            "demographic": {
                "age": "Client age (18-95)",
                "job": "Job type (admin, blue-collar, entrepreneur, etc.)",
                "marital": "Marital status (divorced, married, single, unknown)",
                "education": "Education level (basic.4y, basic.6y, basic.9y, etc.)"
            },
            "financial": {
                "default": "Has credit in default? (yes, no, unknown)",
                "housing": "Has housing loan? (yes, no, unknown)",
                "loan": "Has personal loan? (yes, no, unknown)"
            },
            "campaign": {
                "contact": "Contact type (cellular, telephone)",
                "month": "Last contact month (jan-dec)",
                "day_of_week": "Last contact day (mon-fri)",
                "duration": "Last contact duration in seconds",
                "campaign": "Number of contacts during this campaign",
                "pdays": "Days since last contact (999 if never contacted)",
                "previous": "Number of contacts before this campaign",
                "poutcome": "Previous campaign outcome (failure, nonexistent, success)"
            },
            "economic": {
                "emp_var_rate": "Employment variation rate",
                "cons_price_idx": "Consumer price index",
                "cons_conf_idx": "Consumer confidence index",
                "euribor3m": "Euribor 3 month rate",
                "nr_employed": "Number of employees"
            }
        },
        "total_features": 20
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "api_app:app",
        host="0.0.0.0",
        port=7860,  # Hugging Face Spaces default port
        reload=False
    )