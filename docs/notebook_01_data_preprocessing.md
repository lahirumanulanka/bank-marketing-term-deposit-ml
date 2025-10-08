# Notebook 02: Data Merging & Preprocessing

## 📋 Overview

This notebook implements the critical data preparation phase, where we merge two dataset variants and perform initial preprocessing to create a unified, clean dataset ready for exploratory analysis and modeling.

## 🎯 Learning Objectives

After completing this notebook, you will understand:
1. How to merge datasets with different feature sets
2. Strategies for handling missing values in combined datasets
3. Initial data quality assessment techniques
4. Data type conversions and standardization
5. Best practices for creating reproducible data pipelines

---

## 🔄 Data Merging Strategy

### Challenge: Combining Two Dataset Variants

**Dataset 1**: `bank-full.csv` (45,211 rows, 16 features + target)
- Contains core demographic and campaign features
- Covers May 2008 - November 2010
- Missing: 5 economic/social context features

**Dataset 2**: `bank-additional-full.csv` (41,188 rows, 20 features + target)
- Contains all features including economic indicators
- Same time period with additional context
- Complete feature set

### Merging Approach

#### Option 1: Inner Join (Intersection)
- **Result**: Only rows present in both datasets
- **Size**: Minimal overlap (~0 rows due to different samples)
- **Decision**: ❌ Not suitable - loses most data

#### Option 2: Outer Join (Union)
- **Result**: All rows from both datasets
- **Size**: ~86,399 rows (45,211 + 41,188)
- **Challenge**: Handle missing economic features for bank-full.csv rows
- **Decision**: ✅ Selected approach - maximizes training data

#### Implementation Strategy

```python
# Align columns by adding missing economic features to bank-full.csv
# bank-full.csv gets NaN for: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

# Step 1: Load both datasets
df_full = pd.read_csv('bank-full.csv', sep=';')
df_additional = pd.read_csv('bank-additional-full.csv', sep=';')

# Step 2: Identify missing columns in bank-full
missing_cols = set(df_additional.columns) - set(df_full.columns)

# Step 3: Add missing columns with NaN
for col in missing_cols:
    if col != 'y':  # Don't duplicate target
        df_full[col] = np.nan

# Step 4: Concatenate datasets
df_merged = pd.concat([df_full, df_additional], ignore_index=True)
```

### Rationale for Union Merge

**Benefits**:
1. **Maximized Training Data**: 86,399 samples vs. 41,188
2. **Temporal Coverage**: Full 2008-2010 period preserved
3. **Feature Flexibility**: Can handle missing economic data in preprocessing
4. **Model Robustness**: More diverse training examples

**Handling Missing Economic Features**:
- Strategy 1: Imputation (mean/median for that time period)
- Strategy 2: Create indicator feature (has_economic_data)
- Strategy 3: Use models robust to missing data (tree-based algorithms)
- Selected: Combination of Strategy 2 and 3

---

## 🧹 Initial Data Cleaning

### 1. Duplicate Detection

```python
# Check for exact duplicates
duplicates = df_merged.duplicated().sum()
print(f"Found {duplicates} duplicate rows")

# Remove duplicates
df_merged = df_merged.drop_duplicates()
```

**Findings**:
- Duplicates may exist if same client contacted in both datasets
- Removal ensures each row represents unique campaign contact

### 2. Column Name Standardization

```python
# Standardize column names
df_merged.columns = df_merged.columns.str.replace('.', '_')
df_merged.columns = df_merged.columns.str.lower()
```

**Changes**:
- `emp.var.rate` → `emp_var_rate`
- `cons.price.idx` → `cons_price_idx`
- `cons.conf.idx` → `cons_conf_idx`
- Improves code readability and prevents accessor errors

### 3. Data Type Verification

```python
# Verify data types
print(df_merged.dtypes)

# Convert categorical columns
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 
                   'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y']

for col in categorical_cols:
    df_merged[col] = df_merged[col].astype('category')
```

**Standardized Types**:
- **Numeric**: age, balance, duration, campaign, pdays, previous, economic features
- **Categorical**: job, marital, education, contact types, month, outcome
- **Binary**: default, housing, loan (stored as category)
- **Target**: y (yes/no) - will be label encoded later

---

## 📊 Data Quality Assessment

### Missing Value Analysis

```python
# Calculate missing value statistics
missing_stats = pd.DataFrame({
    'column': df_merged.columns,
    'missing_count': df_merged.isnull().sum(),
    'missing_percentage': (df_merged.isnull().sum() / len(df_merged) * 100).round(2)
})

missing_stats = missing_stats[missing_stats['missing_count'] > 0].sort_values('missing_percentage', ascending=False)
```

**Expected Findings**:

| Feature | Missing Count | Missing % | Source | Handling Strategy |
|---------|--------------|-----------|--------|-------------------|
| emp_var_rate | ~45,211 | 52.3% | Bank-full merge | Create indicator feature + imputation |
| cons_price_idx | ~45,211 | 52.3% | Bank-full merge | Create indicator feature + imputation |
| cons_conf_idx | ~45,211 | 52.3% | Bank-full merge | Create indicator feature + imputation |
| euribor3m | ~45,211 | 52.3% | Bank-full merge | Create indicator feature + imputation |
| nr_employed | ~45,211 | 52.3% | Bank-full merge | Create indicator feature + imputation |

### Missing Value Handling Strategy

#### For Economic Features (from merge):

**Strategy**: Create indicator + impute
```python
# Create indicator feature
df_merged['has_economic_data'] = df_merged['emp_var_rate'].notna().astype(int)

# Impute with temporal mean (grouped by month/year if available)
# For this project, use median imputation
economic_features = ['emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']

for feature in economic_features:
    median_value = df_merged[feature].median()
    df_merged[feature].fillna(median_value, inplace=True)
```

**Rationale**:
- `has_economic_data` captures which samples have complete economic context
- Median imputation prevents distributional distortion
- Tree-based models can learn different patterns for imputed vs. real values

#### For Other Missing Values:

```python
# Check for other missing values in demographic/campaign features
other_missing = df_merged[categorical_cols].isnull().sum()

# Handle unknown categories
# Some datasets use 'unknown' as a category; ensure consistency
```

---

## 🔢 Feature Engineering (Initial)

### 1. Create Economic Data Indicator

```python
# Binary indicator for economic data availability
df_merged['has_economic_data'] = df_merged['emp_var_rate'].notna().astype(int)
```

**Business Justification**:
- Distinguishes samples with full context vs. imputed economic data
- Helps model account for information quality differences
- Useful for temporal analysis (pre/post economic data collection)

### 2. Handle Special Values

```python
# pdays: 999 means "never contacted before"
# Create binary indicator
df_merged['never_contacted_before'] = (df_merged['pdays'] == 999).astype(int)

# Transform pdays for better modeling
df_merged['pdays_transformed'] = df_merged['pdays'].apply(
    lambda x: 0 if x == 999 else x
)
```

**Rationale**:
- pdays=999 is a special code, not a true numeric value
- Separate indicator captures this important information
- Transformed version suitable for distance-based algorithms

---

## 💾 Data Saving Strategy

### Save at Multiple Processing Stages

```python
# Create directory structure
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/interim', exist_ok=True)

# 1. Save merged raw data
df_merged.to_csv('data/raw/bank_merged.csv', index=False)

# 2. Save after initial cleaning (for reproducibility)
df_clean.to_csv('data/interim/bank_cleaned.csv', index=False)
```

**Directory Structure**:
- `data/raw/`: Merged but unprocessed data
- `data/interim/`: After cleaning, before feature engineering
- `data/processed/`: Final feature matrix (created in Notebook 03)

### Data Versioning

```python
# Add processing metadata
metadata = {
    'processing_date': datetime.now().isoformat(),
    'source_files': ['bank-full.csv', 'bank-additional-full.csv'],
    'total_samples': len(df_merged),
    'total_features': len(df_merged.columns),
    'missing_value_strategy': 'median_imputation',
    'duplicate_removal': True
}

# Save metadata
import json
with open('data/interim/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## 📈 Data Overview After Merging

### Merged Dataset Characteristics

```python
# Generate comprehensive summary
print("=" * 50)
print("MERGED DATASET SUMMARY")
print("=" * 50)
print(f"Total Samples: {len(df_merged):,}")
print(f"Total Features: {len(df_merged.columns)}")
print(f"Memory Usage: {df_merged.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nFeature Categories:")
print(f"  - Demographic: 8 features")
print(f"  - Contact Info: 5 features") 
print(f"  - Campaign: 4 features")
print(f"  - Economic: 5 features")
print(f"  - Engineered: 2 features (has_economic_data, never_contacted_before)")
print(f"  - Target: 1 feature (y)")
```

### Sample Data Inspection

```python
# View first few rows
print("\nFirst 5 rows:")
print(df_merged.head())

# View data types
print("\nData Types:")
print(df_merged.dtypes)

# Basic statistics
print("\nNumeric Feature Statistics:")
print(df_merged.describe())
```

---

## ✅ Data Quality Checks

### Validation Checklist

```python
def validate_merged_data(df):
    """Comprehensive data validation"""
    
    checks = {
        'no_duplicates': df.duplicated().sum() == 0,
        'no_completely_missing_rows': df.isnull().all(axis=1).sum() == 0,
        'target_variable_present': 'y' in df.columns,
        'expected_column_count': len(df.columns) >= 21,  # Original 20 + engineered
        'age_range_valid': (df['age'] >= 18).all() and (df['age'] <= 100).all(),
        'balance_plausible': df['balance'].between(-10000, 100000).all(),
        'campaign_positive': (df['campaign'] > 0).all(),
        'target_binary': df['y'].isin(['yes', 'no']).all()
    }
    
    print("Data Quality Validation:")
    for check, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check}: {status}")
    
    return all(checks.values())

# Run validation
is_valid = validate_merged_data(df_merged)
```

### Expected Validation Results

✅ No duplicates after cleaning  
✅ No completely missing rows  
✅ Target variable 'y' present  
✅ 23+ columns (original 21 + 2 engineered)  
✅ Age range reasonable (18-95)  
✅ Balance values plausible  
✅ Campaign contacts > 0  
✅ Target variable binary (yes/no)  

---

## 🔬 Preprocessing Philosophy

### Why This Approach?

1. **Data Preservation**: Union merge retains maximum information
2. **Transparency**: Missing value indicators make imputation visible to models
3. **Reproducibility**: Saved intermediate files enable pipeline debugging
4. **Scalability**: Column alignment strategy works for any number of dataset variants
5. **Model Flexibility**: Preparation compatible with various algorithm types

### Alternative Approaches Considered

#### Approach 1: Use Only bank-additional-full.csv
- ❌ Loses 45,211 samples (~52% of data)
- ❌ Reduces model training capacity
- ✅ No missing values to handle

**Decision**: Rejected due to significant data loss

#### Approach 2: Feature Selection (Drop Economic Features)
- ❌ Loses valuable predictive information
- ❌ Economic indicators shown to improve performance (Moro et al., 2014)
- ✅ Simplifies preprocessing

**Decision**: Rejected due to feature importance

#### Approach 3: Separate Models for Each Dataset
- ❌ Complex deployment (two models)
- ❌ Inconsistent predictions
- ✅ Avoids missing value issue

**Decision**: Rejected due to operational complexity

---

## 📊 Output Summary

### Generated Files

1. **`data/raw/bank_merged.csv`**
   - Merged datasets with aligned columns
   - 86,399 rows × 21+ columns
   - Contains NaN for economic features from bank-full.csv

2. **`data/interim/bank_cleaned.csv`**
   - After initial cleaning and imputation
   - Ready for exploratory analysis
   - Includes engineered indicator features

3. **`data/interim/metadata.json`**
   - Processing metadata
   - Data lineage information
   - Quality check results

### Data Ready For

✅ **Exploratory Data Analysis** (Notebook 03)
- Distribution analysis
- Correlation studies
- Outlier detection
- Feature relationships

✅ **Feature Engineering** (Notebook 03)
- Domain-specific transformations
- Interaction features
- Temporal aggregations

✅ **Model Development** (Notebook 04)
- Algorithm training
- Hyperparameter tuning
- Cross-validation

---

## 🚀 Next Steps

Proceed to **[Notebook 03: Exploratory Data Analysis](notebook_03_exploratory_analysis.md)** to:
- Analyze feature distributions and relationships
- Detect and handle outliers
- Visualize class imbalance
- Engineer domain-specific features
- Prepare final feature matrix for modeling

---

## 📚 Key Takeaways

1. **Data merging requires careful planning** - Consider feature alignment and missing values
2. **Preserve maximum information** - Union merge preferred when datasets complement each other
3. **Create indicator features** - Make data quality differences visible to models
4. **Save intermediate outputs** - Enables debugging and pipeline reproducibility
5. **Validate rigorously** - Automated checks prevent downstream errors

---

**Note**: This preprocessing strategy balances data quantity (86K samples) with data quality (handled missing values systematically). The choice to merge datasets significantly improves model training capacity while maintaining transparency about data limitations through engineered indicator features.
