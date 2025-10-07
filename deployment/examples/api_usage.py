"""
Example: Using the Bank Marketing Prediction API

This script demonstrates how to make predictions using the FastAPI REST API.
"""

import requests
import json

# API endpoint (update with your actual endpoint)
API_URL = "http://localhost:8000"

def test_health():
    """Test API health check"""
    print("=" * 60)
    print("Testing Health Check...")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_model_info():
    """Get model information"""
    print("=" * 60)
    print("Getting Model Information...")
    print("=" * 60)
    
    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def make_prediction(client_data):
    """Make a prediction"""
    print("=" * 60)
    print("Making Prediction...")
    print("=" * 60)
    print(f"Input: {json.dumps(client_data, indent=2)}")
    print()
    
    response = requests.post(
        f"{API_URL}/predict",
        json=client_data
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print()
        print(f"✅ Prediction: {result['prediction'].upper()}")
        print(f"📊 Probability: {result['probability']:.2%}")
        print(f"🎯 Confidence: {result['confidence'].upper()}")
    else:
        print(f"❌ Error: {response.text}")
    print()

# Example 1: Likely to subscribe (high probability)
client_1 = {
    "age": 32,
    "job": "management",
    "marital": "single",
    "education": "university.degree",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "duration": 500,  # Long call duration
    "campaign": 1,  # First contact
    "pdays": 999,  # Not previously contacted
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.1,
    "cons_price_idx": 93.994,
    "cons_conf_idx": -36.4,
    "euribor3m": 4.857,
    "nr_employed": 5191.0
}

# Example 2: Unlikely to subscribe (low probability)
client_2 = {
    "age": 60,
    "job": "retired",
    "marital": "married",
    "education": "basic.4y",
    "default": "no",
    "housing": "yes",
    "loan": "yes",
    "contact": "telephone",
    "month": "nov",
    "day_of_week": "fri",
    "duration": 50,  # Short call duration
    "campaign": 10,  # Many contacts
    "pdays": 5,  # Recently contacted
    "previous": 3,
    "poutcome": "failure",
    "emp_var_rate": -1.8,
    "cons_price_idx": 92.893,
    "cons_conf_idx": -46.2,
    "euribor3m": 1.313,
    "nr_employed": 5099.1
}

# Example 3: Moderate probability
client_3 = {
    "age": 35,
    "job": "technician",
    "marital": "married",
    "education": "high.school",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "jul",
    "day_of_week": "wed",
    "duration": 250,
    "campaign": 2,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "emp_var_rate": 1.4,
    "cons_price_idx": 93.918,
    "cons_conf_idx": -42.7,
    "euribor3m": 4.961,
    "nr_employed": 5228.1
}

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BANK MARKETING PREDICTION API - EXAMPLES")
    print("=" * 60 + "\n")
    
    # Test health
    try:
        test_health()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Make sure the API is running: python deployment/api.py")
        exit(1)
    
    # Test model info
    try:
        test_model_info()
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    # Make predictions
    print("📊 Example 1: High Probability Client")
    print("-" * 60)
    make_prediction(client_1)
    
    print("📊 Example 2: Low Probability Client")
    print("-" * 60)
    make_prediction(client_2)
    
    print("📊 Example 3: Moderate Probability Client")
    print("-" * 60)
    make_prediction(client_3)
    
    print("=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    
    # Batch prediction example
    print("\n" + "=" * 60)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 60 + "\n")
    
    clients = [client_1, client_2, client_3]
    results = []
    
    for i, client in enumerate(clients, 1):
        response = requests.post(f"{API_URL}/predict", json=client)
        if response.status_code == 200:
            result = response.json()
            results.append({
                'client': i,
                'prediction': result['prediction'],
                'probability': result['probability']
            })
    
    print("Batch Results:")
    print("-" * 60)
    for r in results:
        print(f"Client {r['client']}: {r['prediction'].upper()} ({r['probability']:.2%})")
    print()
