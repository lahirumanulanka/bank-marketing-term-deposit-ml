"""
Bank Marketing Term Deposit Prediction - HuggingFace Spaces App
Gradio interface for predicting term deposit subscription
"""

import gradio as gr
import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Load model and preprocessors
MODEL_PATH = "xgboost_retrained_tuned.pkl"
SCALER_PATH = "preprocessing/scaler.pkl"
ENCODERS_PATH = "preprocessing/label_encoders.pkl"

def load_model():
    """Load the trained model and preprocessors"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(ENCODERS_PATH, 'rb') as f:
            encoders = pickle.load(f)
        return model, scaler, encoders
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

model, scaler, encoders = load_model()

def predict_subscription(age, job, marital, education, default, housing, loan, 
                         contact, month, day_of_week, duration, campaign, pdays, 
                         previous, poutcome, emp_var_rate, cons_price_idx, 
                         cons_conf_idx, euribor3m, nr_employed):
    """
    Predict whether a client will subscribe to a term deposit
    
    Returns:
        - Prediction (Yes/No)
        - Probability
        - Confidence interpretation
    """
    
    if model is None:
        return "Error: Model not loaded", 0.0, "Model not available"
    
    # Create input dataframe
    # Create input dataframe with basic features
    input_data = pd.DataFrame({
        'age': [int(age)],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [int(duration)],
        'campaign': [int(campaign)],
        'pdays': [int(pdays)],
        'previous': [int(previous)],
        'poutcome': [poutcome],
        'emp.var.rate': [float(emp_var_rate)],
        'cons.price.idx': [float(cons_price_idx)],
        'cons.conf.idx': [float(cons_conf_idx)],
        'euribor3m': [float(euribor3m)],
        'nr.employed': [float(nr_employed)]
    })
    
    # Add engineered features that the model expects
    input_data['balance'] = 0  # Default balance (not provided in input)
    input_data['day'] = 15     # Default day of month (not provided in input)
    input_data['data_source'] = 'gradio'  # Source identifier
    
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
    
    # Scale numerical features (only the basic ones that scaler was fitted on)
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
    
    # Reorder columns to match expected features and ensure all are numeric
    input_data = input_data.reindex(columns=expected_features, fill_value=0)
    
    # Convert all columns to numeric to avoid XGBoost dtype issues
    for col in input_data.columns:
        input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Interpret results
    result = "Yes" if prediction == 1 else "No"
    prob_yes = probability[1] * 100
    
    if prob_yes >= 80:
        confidence = "Very High"
    elif prob_yes >= 70:
        confidence = "High"
    elif prob_yes >= 60:
        confidence = "Moderate"
    elif prob_yes >= 50:
        confidence = "Low"
    else:
        confidence = "Very Low"
    
    return result, f"{prob_yes:.2f}%", confidence

# Create Gradio interface
with gr.Blocks(title="Bank Marketing Term Deposit Prediction") as demo:
    gr.Markdown(
        """
        # 🏦 Bank Marketing Term Deposit Prediction
        
        Predict whether a client will subscribe to a term deposit based on marketing campaign data.
        
        **Model**: XGBoost Classifier (Trained on UCI Bank Marketing Dataset)
        **Features**: 27 engineered features from 20 input parameters
        **Performance**: ROC-AUC ~0.93 with high prediction accuracy
        """
    )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 👤 Client Information")
            age = gr.Slider(18, 95, value=30, label="Age", step=1)
            job = gr.Dropdown(
                choices=["admin.", "blue-collar", "entrepreneur", "housemaid", 
                        "management", "retired", "self-employed", "services", 
                        "student", "technician", "unemployed", "unknown"],
                value="admin.",
                label="Job"
            )
            marital = gr.Radio(
                choices=["divorced", "married", "single", "unknown"],
                value="single",
                label="Marital Status"
            )
            education = gr.Dropdown(
                choices=["basic.4y", "basic.6y", "basic.9y", "high.school", 
                        "illiterate", "professional.course", "university.degree", "unknown"],
                value="university.degree",
                label="Education"
            )
            
        with gr.Column():
            gr.Markdown("### 💳 Financial Information")
            default = gr.Radio(
                choices=["no", "yes", "unknown"],
                value="no",
                label="Has Credit in Default?"
            )
            housing = gr.Radio(
                choices=["no", "yes", "unknown"],
                value="no",
                label="Has Housing Loan?"
            )
            loan = gr.Radio(
                choices=["no", "yes", "unknown"],
                value="no",
                label="Has Personal Loan?"
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📞 Campaign Information")
            contact = gr.Radio(
                choices=["cellular", "telephone"],
                value="cellular",
                label="Contact Type"
            )
            month = gr.Dropdown(
                choices=["jan", "feb", "mar", "apr", "may", "jun", 
                        "jul", "aug", "sep", "oct", "nov", "dec"],
                value="may",
                label="Contact Month"
            )
            day_of_week = gr.Dropdown(
                choices=["mon", "tue", "wed", "thu", "fri"],
                value="mon",
                label="Contact Day of Week"
            )
            duration = gr.Slider(0, 5000, value=180, label="Last Contact Duration (seconds)", step=1)
            
        with gr.Column():
            gr.Markdown("### 📊 Campaign Metrics")
            campaign = gr.Slider(1, 50, value=1, label="Number of Contacts (this campaign)", step=1)
            pdays = gr.Slider(0, 999, value=999, label="Days Since Last Contact (999 = not contacted)", step=1)
            previous = gr.Slider(0, 10, value=0, label="Number of Contacts (previous campaigns)", step=1)
            poutcome = gr.Dropdown(
                choices=["failure", "nonexistent", "success"],
                value="nonexistent",
                label="Previous Campaign Outcome"
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📈 Economic Indicators")
            emp_var_rate = gr.Number(value=1.1, label="Employment Variation Rate")
            cons_price_idx = gr.Number(value=93.994, label="Consumer Price Index")
            cons_conf_idx = gr.Number(value=-36.4, label="Consumer Confidence Index")
            euribor3m = gr.Number(value=4.857, label="Euribor 3 Month Rate")
            nr_employed = gr.Number(value=5191.0, label="Number of Employees")
    
    with gr.Row():
        predict_btn = gr.Button("🔮 Predict Subscription", variant="primary", size="lg")
    
    with gr.Row():
        with gr.Column():
            output_prediction = gr.Textbox(label="Prediction")
            output_probability = gr.Textbox(label="Probability")
            output_confidence = gr.Textbox(label="Confidence Level")
    
    # Connect prediction function
    predict_btn.click(
        fn=predict_subscription,
        inputs=[age, job, marital, education, default, housing, loan, contact, 
                month, day_of_week, duration, campaign, pdays, previous, poutcome,
                emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed],
        outputs=[output_prediction, output_probability, output_confidence]
    )
    
    gr.Markdown(
        """
        ---
        ### 📝 About
        
        This model predicts term deposit subscription using an **XGBoost classifier** trained on the 
        UCI Bank Marketing Dataset (Portuguese banking institution, 2008-2010).
        
        **Key Features:**
        - Trained on 86,000+ samples with SMOTE for class imbalance
        - Uses 20 input features → 27 engineered features
        - Advanced feature engineering including age groups, duration categories
        - Economic indicators and campaign history analysis
        
        **Model Performance:**
        - Algorithm: XGBoost Classifier
        - ROC-AUC: ~0.93
        - High precision with confidence-based predictions
        - Real-time feature engineering pipeline
        
        **Confidence Levels:**
        - Very High: ≥80% probability
        - High: 70-79% probability  
        - Moderate: 60-69% probability
        - Low: 50-59% probability
        - Very Low: <50% probability
        
        **Author**: Lahiru Manulanka Munasinghe  
        **Repository**: [GitHub](https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml)
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
