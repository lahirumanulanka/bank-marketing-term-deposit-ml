---
title: Bank Marketing Term Deposit Prediction
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: mit
---

# Bank Marketing Term Deposit Prediction

Predict whether a client will subscribe to a term deposit based on direct marketing campaign data from a Portuguese banking institution.

## Model Information

- **Model Type**: LightGBM Classifier
- **Dataset**: UCI Bank Marketing Dataset (86,000+ samples)
- **Features**: 20 input features including demographic, financial, and economic indicators
- **Performance**: ROC-AUC ~0.93

## Features

The model uses the following input features:

### Client Information
- Age
- Job type
- Marital status
- Education level

### Financial Status
- Credit in default
- Housing loan
- Personal loan

### Campaign Details
- Contact type (cellular/telephone)
- Last contact month and day of week
- Last contact duration
- Number of contacts in current campaign
- Days since last contact
- Number of contacts in previous campaigns
- Previous campaign outcome

### Economic Indicators
- Employment variation rate
- Consumer price index
- Consumer confidence index
- Euribor 3 month rate
- Number of employees

## Usage

1. Fill in the client and campaign information
2. Click "Predict Subscription"
3. Get prediction with probability and confidence level

## Model Training

The model was trained using:
- SMOTE for class imbalance handling
- Hyperparameter tuning with cross-validation
- Feature engineering based on domain knowledge
- MLflow for experiment tracking

## Repository

Full project code and documentation: [GitHub Repository](https://github.com/lahirumanulanka/bank-marketing-term-deposit-ml)

## Author

Lahiru Manulanka Munasinghe

## License

MIT License
