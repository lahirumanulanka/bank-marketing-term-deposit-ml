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
MODEL_PATH = "lightgbm_retrained_tuned.pkl"
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
    
    # Encode categorical features
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome']
    
    for col in categorical_cols:
        if col in encoders:
            # Handle unknown categories
            if input_data[col].iloc[0] not in encoders[col].classes_:
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
    result = "Yes" if prediction == 1 else "No"
    prob_yes = probability[1] * 100
    
    if prob_yes >= 70:
        confidence = "High confidence"
    elif prob_yes >= 50:
        confidence = "Moderate confidence"
    else:
        confidence = "Low confidence"
    
    return result, f"{prob_yes:.2f}%", confidence

# Create Gradio interface
with gr.Blocks(title="Bank Marketing Term Deposit Prediction") as demo:
    gr.Markdown(
        """
        # 🏦 Bank Marketing Term Deposit Prediction
        
        Predict whether a client will subscribe to a term deposit based on marketing campaign data.
        
        **Model**: LightGBM (Trained on UCI Bank Marketing Dataset)
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
        
        This model predicts term deposit subscription using a LightGBM classifier trained on the 
        UCI Bank Marketing Dataset (Portuguese banking institution, 2008-2010).
        
        **Key Features:**
        - Trained on 86,000+ samples
        - Handles class imbalance with SMOTE
        - Uses 20 input features including economic indicators
        
        **Model Performance:**
        - ROC-AUC: ~0.93
        - Precision/Recall optimized for banking context
        
        **Author**: Lahiru Manulanka Munasinghe  
        **Repository**: [GitHub](https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml)
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
