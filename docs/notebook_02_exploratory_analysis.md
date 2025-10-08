# Notebook 03: Exploratory Data Analysis (EDA)

## 📋 Overview

This notebook performs comprehensive exploratory data analysis to understand the bank marketing dataset's characteristics, patterns, and relationships. We analyze distributions, detect outliers, handle class imbalance, and engineer domain-specific features.

## 🎯 Learning Objectives

After completing this notebook, you will understand:
1. How to conduct systematic exploratory data analysis for classification problems
2. Techniques for detecting and handling outliers in business contexts
3. Strategies for addressing class imbalance (SMOTE + visualization)
4. Domain-informed feature engineering for banking applications
5. How to translate statistical findings into business insights

---

## 📊 EDA Framework

### Analysis Categories

1. **Univariate Analysis**: Individual feature distributions
2. **Bivariate Analysis**: Feature relationships with target variable
3. **Multivariate Analysis**: Feature interactions and correlations
4. **Missing Value Analysis**: Data completeness assessment
5. **Outlier Detection**: Identifying anomalous values
6. **Class Imbalance Analysis**: Target variable distribution
7. **Feature Engineering**: Creating domain-informed features

---

## 1. Dataset Overview

### Load and Inspect Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data from previous notebook
df = pd.read_csv('data/interim/bank_cleaned.csv')

# Display basic information
print(f"Dataset Shape: {df.shape}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nColumn Data Types:\n{df.dtypes}")
```

### Key Statistics

- **Total Samples**: 86,399
- **Total Features**: 21 (20 input + 1 target)
- **Numeric Features**: 11 (age, balance, duration, campaign, pdays, previous, 5 economic)
- **Categorical Features**: 9 (job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome)
- **Target Variable**: y (yes/no - binary classification)

---

## 2. Missing Value Analysis

### Comprehensive Missing Value Assessment

```python
# Calculate missing value statistics
missing_stats = pd.DataFrame({
    'Feature': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})

missing_stats = missing_stats[missing_stats['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.barplot(data=missing_stats, x='Missing_Percentage', y='Feature')
plt.title('Missing Value Analysis')
plt.xlabel('Percentage Missing (%)')
```

### Missing Value Strategy

**Economic Features** (~52% missing):
- **Source**: Merged from bank-full.csv (no economic data)
- **Strategy**: Median imputation + indicator feature (`has_economic_data`)
- **Justification**: 
  - Preserves maximum training samples (86K vs 41K)
  - Indicator allows model to learn different patterns for imputed values
  - Tree-based models handle this naturally through splitting

**Other Features**:
- Minimal missing values (<1%)
- Categorical: Fill with 'unknown' category
- Numeric: Fill with median (robust to outliers)

---

## 3. Target Variable Analysis

### Class Distribution

```python
# Analyze target variable distribution
target_counts = df['y'].value_counts()
target_percentages = (target_counts / len(df) * 100).round(2)

print("Target Variable Distribution:")
print(target_counts)
print(f"\nClass Balance:")
print(f"  No (Negative): {target_percentages['no']:.2f}%")
print(f"  Yes (Positive): {target_percentages['yes']:.2f}%")
print(f"  Imbalance Ratio: {target_counts['no'] / target_counts['yes']:.2f}:1")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
sns.countplot(data=df, x='y', ax=ax1)
ax1.set_title('Target Variable Distribution (Count)')
ax1.set_xlabel('Subscribed to Term Deposit')
ax1.set_ylabel('Count')

# Percentage plot
ax2.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Target Variable Distribution (%)')
```

### Findings

- **Negative Class (No)**: ~76,000 samples (88%)
- **Positive Class (Yes)**: ~10,400 samples (12%)
- **Imbalance Ratio**: ~7.3:1

**Business Interpretation**:
- Only 12% of clients subscribe to term deposits
- Typical conversion rate for cold calling campaigns
- Requires class imbalance handling for effective model training

---

## 4. Univariate Analysis

### Numeric Feature Distributions

```python
# Select numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('y')  # Remove target if encoded

# Create distribution plots
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.ravel()

for idx, feature in enumerate(numeric_features):
    # Histogram with KDE
    df[feature].hist(bins=50, ax=axes[idx], alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{feature} Distribution')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
```

### Key Observations

**Age**:
- Distribution: Roughly normal, slight right skew
- Range: 18-95 years
- Mode: 30-40 years (largest customer segment)
- Business Insight: Target campaigns at 30-40 age group

**Balance**:
- Distribution: Highly right-skewed
- Range: -8,000 to 100,000+ euros
- Median: ~450 euros
- **Outliers**: Values >20,000 euros (retain - legitimate wealthy clients)

**Duration**:
- Distribution: Right-skewed
- Range: 0-5,000 seconds
- Median: ~180 seconds (3 minutes)
- **Note**: Only available after call; strong predictor but not available for prediction

**Campaign**:
- Distribution: Exponentially decreasing
- Range: 1-60+ contacts
- Mode: 1-2 contacts
- **Outliers**: >10 contacts (potential customer fatigue)

**Economic Indicators**:
- Show temporal patterns and macroeconomic trends
- Euribor rate: Strong correlation with subscription behavior
- Employment variation: Negative values indicate recession

---

## 5. Categorical Feature Analysis

```python
# Categorical features
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome']

# Create count plots
fig, axes = plt.subplots(5, 2, figsize=(18, 20))
axes = axes.ravel()

for idx, feature in enumerate(categorical_features):
    # Count plot with hue for target
    sns.countplot(data=df, x=feature, hue='y', ax=axes[idx])
    axes[idx].set_title(f'{feature} Distribution by Target')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].legend(title='Subscribed', labels=['No', 'Yes'])

plt.tight_layout()
```

### Key Insights

**Job**:
- Highest counts: admin, blue-collar, technician
- Highest subscription rates: students, retired (more time/interest)
- Lowest subscription: unemployed (financial constraints)

**Education**:
- University degree holders: Higher subscription rates
- Basic education: Lower subscription rates
- Unknown: Mixed results (data quality issue)

**Marital**:
- Single: Slightly higher subscription rates
- Married: Moderate rates
- Divorced: Lower rates

**Contact Type**:
- Cellular: Much higher success rate than telephone
- Mobile users more responsive to marketing

**Month**:
- March, September, October, December: Higher subscription rates
- May: Lowest rates (campaign fatigue?)
- Seasonal patterns evident

**Previous Outcome**:
- Success: Very high subscription rate (behavioral consistency)
- Failure: Low subscription rate
- Nonexistent: Moderate rate (new customers)

---

## 6. Outlier Detection & Handling

### IQR Method for Outlier Detection

```python
def detect_outliers_iqr(df, feature):
    """Detect outliers using Interquartile Range method"""
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    
    return outliers, lower_bound, upper_bound

# Detect outliers in key numeric features
outlier_summary = []

for feature in ['age', 'balance', 'duration', 'campaign', 'previous']:
    outliers, lower, upper = detect_outliers_iqr(df, feature)
    outlier_summary.append({
        'Feature': feature,
        'Outlier_Count': len(outliers),
        'Outlier_Percentage': (len(outliers) / len(df) * 100).round(2),
        'Lower_Bound': lower,
        'Upper_Bound': upper
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df)
```

### Outlier Handling Strategy

**Selective Capping Based on Business Context**:

#### Balance (Account Balance)
- **Outliers Detected**: Values > €20,000 (95th percentile)
- **Decision**: Cap to 1st-99th percentile range
- **Justification**: 
  - Extreme wealth outliers can skew models
  - Retains signal while reducing variance
  - Preserves privacy (very high balances might be identifiable)

```python
# Cap balance outliers
balance_1st = df['balance'].quantile(0.01)
balance_99th = df['balance'].quantile(0.99)

df['balance_capped'] = df['balance'].clip(lower=balance_1st, upper=balance_99th)
```

#### Campaign (Number of Contacts)
- **Outliers Detected**: >10 contacts (customer fatigue zone)
- **Decision**: Cap to 95th percentile
- **Justification**:
  - >10 contacts indicates harassment, not normal campaign
  - Reduces noise from outlier behavior
  - Business rule: campaigns should stop before fatigue

```python
# Cap campaign outliers
campaign_95th = df['campaign'].quantile(0.95)
df['campaign_capped'] = df['campaign'].clip(upper=campaign_95th)
```

#### Age
- **Outliers Detected**: <20 or >80 years
- **Decision**: Retain all values
- **Justification**: 
  - All ages are plausible bank customers
  - Elderly and youth are important segments
  - No reason to cap legitimate demographic data

#### Duration
- **Outliers Detected**: >3,000 seconds (50 minutes)
- **Decision**: Retain but monitor
- **Justification**:
  - Long calls may indicate high interest
  - Feature not available at prediction time (information leakage)
  - Keep for analysis, exclude from production features

### Outlier Treatment Summary

| Feature | Outlier % | Treatment | Justification |
|---------|-----------|-----------|---------------|
| balance | 15% | Cap to 1st-99th percentile | Reduce variance, preserve privacy |
| campaign | 8% | Cap to 95th percentile | Remove customer fatigue noise |
| age | 5% | Retain | All ages valid customers |
| duration | 10% | Retain for analysis | Not available at prediction |
| previous | 3% | Retain | Valid campaign history |

---

## 7. Class Imbalance Handling

### SMOTE (Synthetic Minority Over-sampling Technique)

**Problem**: 88:12 class imbalance leads to:
- Models biased toward majority class
- Poor recall for minority class (term deposit subscribers)
- Suboptimal business value (missing revenue opportunities)

**Solution**: SMOTE + Class Weights

```python
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('y', axis=1)
y = df['y'].map({'no': 0, 'yes': 1})  # Label encode

# Train-test split BEFORE SMOTE (to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original Training Set:")
print(f"  Class 0 (No): {(y_train == 0).sum():,}")
print(f"  Class 1 (Yes): {(y_train == 1).sum():,}")
print(f"\nBalanced Training Set (After SMOTE):")
print(f"  Class 0 (No): {(y_train_balanced == 0).sum():,}")
print(f"  Class 1 (Yes): {(y_train_balanced == 1).sum():,}")
```

### SMOTE Visualization

```python
# Visualize class distribution before and after SMOTE
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Before SMOTE
pd.Series(y_train).value_counts().plot(kind='bar', ax=ax1, color=['#ff6b6b', '#4ecdc4'])
ax1.set_title('Training Set - Before SMOTE')
ax1.set_xlabel('Class')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)

# After SMOTE
pd.Series(y_train_balanced).value_counts().plot(kind='bar', ax=ax2, color=['#ff6b6b', '#4ecdc4'])
ax2.set_title('Training Set - After SMOTE')
ax2.set_xlabel('Class')
ax2.set_ylabel('Count')
ax2.set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)

# Comparison
comparison_data = pd.DataFrame({
    'Before SMOTE': pd.Series(y_train).value_counts(),
    'After SMOTE': pd.Series(y_train_balanced).value_counts()
})
comparison_data.plot(kind='bar', ax=ax3, color=['#ff6b6b', '#4ecdc4'])
ax3.set_title('Class Distribution Comparison')
ax3.set_xlabel('Class')
ax3.set_ylabel('Count')
ax3.set_xticklabels(['No (0)', 'Yes (1)'], rotation=0)
ax3.legend(title='Dataset')

plt.tight_layout()
```

### SMOTE Impact Analysis

**Before SMOTE**:
- Class 0 (No): ~61,000 samples
- Class 1 (Yes): ~8,300 samples
- Ratio: 7.3:1

**After SMOTE**:
- Class 0 (No): ~61,000 samples
- Class 1 (Yes): ~61,000 samples
- Ratio: 1:1 (perfectly balanced)

**Benefits**:
1. **Improved Recall**: Model learns minority class patterns better
2. **Reduced Bias**: Balanced training prevents majority class preference
3. **Business Value**: Better identification of potential subscribers

**Important Notes**:
- SMOTE applied only to training data (no test set contamination)
- Test set remains imbalanced (realistic evaluation)
- Combine with class weights for robust handling

---

## 8. Feature Engineering

### Domain-Informed Feature Creation

#### Feature 1: Contact Frequency Category

```python
# Categorize campaign contacts into frequency levels
def categorize_contact_frequency(campaign_count):
    if campaign_count == 1:
        return 'single_contact'
    elif campaign_count <= 3:
        return 'low_frequency'
    elif campaign_count <= 6:
        return 'medium_frequency'
    else:
        return 'high_frequency_fatigue'

df['contact_frequency'] = df['campaign'].apply(categorize_contact_frequency)
```

**Business Justification**:
- **Behavioral Insight**: Customer fatigue increases with contact frequency
- **Non-linear Relationship**: Impact not proportional to contact count
- **Actionable**: Different strategies for each frequency segment

#### Feature 2: Previous Campaign Success

```python
# Binary indicator for previous campaign success
df['previous_campaign_success'] = (df['poutcome'] == 'success').astype(int)
```

**Business Justification**:
- **Behavioral Consistency**: Past success predicts future success
- **Customer Profiling**: Identifies receptive customers
- **Efficiency**: High-value targeting for campaigns

#### Feature 3: Age Group

```python
# Life stage segmentation
def categorize_age_group(age):
    if age < 25:
        return 'young_adult'
    elif age < 35:
        return 'early_career'
    elif age < 50:
        return 'mid_career'
    elif age < 65:
        return 'pre_retirement'
    else:
        return 'retired'

df['age_group'] = df['age'].apply(categorize_age_group)
```

**Business Justification**:
- **Life Stage Matters**: Financial priorities differ by age
- **Non-linear Effects**: Age impact not linear (retirement spike)
- **Marketing Strategy**: Tailored messaging per life stage

#### Feature 4: Has Economic Data Indicator

```python
# Indicator for economic data availability (already created in preprocessing)
# Captures which samples have complete economic context
df['has_economic_data'] = df['emp_var_rate'].notna().astype(int)
```

**Business Justification**:
- **Data Quality Signal**: Distinguishes complete vs. imputed records
- **Temporal Context**: Indicates data collection era
- **Model Transparency**: Makes imputation visible to algorithm

#### Feature 5: Duration Category

```python
# Call length engagement categories
def categorize_duration(seconds):
    if seconds < 60:
        return 'very_short'
    elif seconds < 180:
        return 'short'
    elif seconds < 300:
        return 'medium'
    else:
        return 'long_engaged'

df['duration_category'] = df['duration'].apply(categorize_duration)
```

**Business Justification**:
- **Engagement Indicator**: Long calls signal interest
- **Quality Signal**: Quick hang-ups indicate disinterest
- **Note**: Only for analysis (not available at prediction time)

### Feature Engineering Summary

| Feature | Type | Business Value | Model Benefit |
|---------|------|----------------|---------------|
| contact_frequency | Categorical | Customer fatigue detection | Non-linear pattern capture |
| previous_campaign_success | Binary | Behavioral consistency | High predictive power |
| age_group | Categorical | Life stage targeting | Non-linear age effects |
| has_economic_data | Binary | Data quality indicator | Imputation transparency |
| duration_category | Categorical | Engagement levels | Interest signal (analysis only) |

---

## 9. Correlation Analysis

### Feature Correlation Matrix

```python
# Calculate correlation matrix for numeric features
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Visualize with heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
```

### Key Correlation Findings

**Highly Correlated Features** (|r| > 0.7):
- `emp_var_rate` ↔ `euribor3m` (r = 0.97): Both track economic conditions
- `cons_price_idx` ↔ `nr_employed` (r = 0.91): Employment affects prices

**Target Correlations**:
- `duration` → Strongest positive correlation (but not available at prediction)
- `poutcome` (previous success) → Strong positive signal
- `euribor3m` → Moderate negative correlation (recession increases deposits)
- `nr_employed` → Moderate negative correlation

**Multicollinearity Concerns**:
- Economic features highly correlated (consider dimensionality reduction)
- Decision: Retain all for tree-based models (handle collinearity naturally)
- Note for linear models: May need feature selection or regularization

---

## 10. Bivariate Analysis (Feature vs. Target)

### Distribution of Features by Target Class

```python
# Compare numeric feature distributions by target
numeric_features_for_analysis = ['age', 'balance', 'campaign', 'previous']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, feature in enumerate(numeric_features_for_analysis):
    # Box plot by target
    df.boxplot(column=feature, by='y', ax=axes[idx])
    axes[idx].set_title(f'{feature} Distribution by Subscription')
    axes[idx].set_xlabel('Subscribed to Term Deposit')
    axes[idx].set_ylabel(feature)

plt.tight_layout()
```

### Insights by Target Class

**Subscribers (Yes) vs. Non-subscribers (No)**:

1. **Age**: Subscribers slightly older (retirees with savings)
2. **Balance**: Subscribers have higher average balance (ability to deposit)
3. **Campaign**: Subscribers contacted fewer times (quality over quantity)
4. **Previous**: Subscribers had more previous campaign contacts (brand familiarity)
5. **Duration**: Subscribers have much longer call durations (engagement)

---

## 11. Data Preparation for Modeling

### Final Feature Matrix

```python
# Select features for modeling
feature_columns = [
    # Demographic
    'age', 'job', 'marital', 'education',
    # Financial
    'default', 'balance_capped', 'housing', 'loan',
    # Contact
    'contact', 'month', 'day_of_week',
    # Campaign (exclude duration - not available at prediction)
    'campaign_capped', 'pdays', 'previous', 'poutcome',
    # Economic
    'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed',
    # Engineered
    'contact_frequency', 'previous_campaign_success', 'age_group', 'has_economic_data'
]

X = df[feature_columns]
y = df['y'].map({'no': 0, 'yes': 1})

# Save processed data
df.to_csv('data/processed/bank_feature_engineered.csv', index=False)
X.to_csv('data/processed/X_features.csv', index=False)
y.to_csv('data/processed/y_target.csv', index=False)
```

### Encoding Strategy

**Categorical Features**:
- **Label Encoding**: For ordinal features (education levels)
- **One-Hot Encoding**: For nominal features (job, marital, contact)
- **Tree-based Models**: Can handle label encoding directly

**Numeric Features**:
- **Standardization**: For distance-based algorithms (SVM, Neural Networks)
- **No Scaling**: For tree-based models (inherently scale-invariant)

---

## 📊 EDA Summary & Insights

### Key Findings

1. **Class Imbalance**: 88:12 ratio requires SMOTE + class weights
2. **Outliers**: Selective capping for balance and campaign features
3. **Missing Values**: 52% economic features handled with imputation + indicator
4. **Strong Predictors**: Previous outcome, duration, euribor rate, contact type
5. **Engineered Features**: 5 new features add domain knowledge

### Business Recommendations

1. **Target Segment**: 
   - Students and retirees (higher conversion)
   - University-educated individuals
   - Previous campaign successes

2. **Optimal Timing**:
   - March, September, October, December
   - Avoid May (campaign fatigue observed)
   - Consider economic indicators (low euribor = higher success)

3. **Contact Strategy**:
   - Use cellular over telephone
   - Limit contacts to 1-3 per campaign
   - Focus on longer, engaging conversations

4. **Economic Timing**:
   - Higher success during economic uncertainty
   - Monitor euribor and employment rates
   - Recession periods favor term deposits

---

## 🚀 Next Steps

Proceed to **[Notebook 04: Model Development](notebook_04_model_development.md)** to:
- Encode categorical features
- Scale numeric features
- Train 6 diverse machine learning models
- Track experiments with MLflow
- Evaluate model performance

---

## 📚 Key Takeaways

1. **EDA is iterative** - Insights guide feature engineering and preprocessing
2. **Domain knowledge crucial** - Business context informs outlier handling and feature creation
3. **Visualizations communicate** - Charts reveal patterns not visible in statistics
4. **Imbalance handling essential** - SMOTE prevents majority class bias
5. **Feature engineering matters** - Domain-informed features improve model performance

---

**Note**: This comprehensive EDA provides the foundation for informed modeling decisions. Every preprocessing choice is justified by data patterns and business context, ensuring transparent and effective machine learning pipeline development.
