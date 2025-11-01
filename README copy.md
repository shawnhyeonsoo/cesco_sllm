# Churn Prediction Model - Final Package
Generated: 2025-11-01 16:44:12

## 📁 Folder Structure

```
churn_model_final/
├── data/
│   ├── train_data.csv          # Training dataset (14,933 samples)
│   └── test_data.csv           # Test dataset (3,734 samples)
├── models/
│   └── churn_model.pkl         # Trained Random Forest model
└── results/
    ├── inference_results.csv   # All predictions with per-user top features
    └── high_risk_customers.csv # Customers with >70% churn probability
```

## 📊 Model Performance

- **Test Accuracy**: 96%
- **Test ROC-AUC**: 0.9568
- **Train ROC-AUC**: 0.9822
- **No Overfitting**: Train-Test difference only 0.0253

## 🎯 Top 3 Globally Most Important Features

1. **work_count** (0.0918)
2. **recency_combined** (0.0564)
3. **work_last_180_days** (0.0557)

## 📈 Dataset Statistics

- **Total Samples**: 18,667
- **Churned**: 2,410 (12.91%)
- **Not Churned**: 16,257 (87.09%)
- **Train Set**: 14,933 (80%)
- **Test Set**: 3,734 (20%)

## 🚨 High-Risk Customers

- **Total High-Risk**: 1,848 customers (probability >= 70%)
- **Average Probability**: 92.5%

## 📝 File Descriptions

### data/train_data.csv
Training dataset with:
- Customer ID (고객코드)
- Customer Name (고객명)
- 100 features
- Target label (해약여부)

### data/test_data.csv
Test dataset (same structure as train_data.csv)

### models/churn_model.pkl
Pickled model containing:
- Trained RandomForestClassifier
- Feature list
- Performance metrics
- Training metadata

### results/inference_results.csv
Predictions for all customers with **personalized top 3 features**:
- 고객코드: Customer ID
- 고객명: Customer Name
- actual_churn: Actual churn status (0/1)
- predicted_churn: Predicted churn status (0/1)
- churn_probability: Probability of churn (0.0-1.0)
- top_feature_1/2/3: Names of THIS customer's top 3 contributing features
- top_feature_1/2/3_value: Values of those features for this customer
- top_feature_1/2/3_contribution: Feature importance × feature value (contribution score)
- **top_feature_1/2/3_description: Human-readable explanation of what the feature means**
- risk_category: Low/Medium/High Risk

**Note**: Each customer has their own unique top 3 features based on their specific profile.

**Example interpretations:**
- "work_days_since_first = 1095" → "Days since first work order (higher = longer relationship)" = 3 years of relationship
- "contract_total_duration = 730" → "Total contract duration in days" = 2 years total
- "work_last_30_days = 0" → "Work orders in last 30 days" = No recent activity (risk signal!)

### results/high_risk_customers.csv
Filtered list of customers with churn probability >= 70%

## 🔧 How to Use

### Load the Model
```python
import pickle
with open('models/churn_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']
```

### Load Data
```python
import pandas as pd
train_df = pd.read_csv('data/train_data.csv')
test_df = pd.read_csv('data/test_data.csv')
```

### Make Predictions
```python
results = pd.read_csv('results/inference_results.csv')
high_risk = results[results['risk_category'] == 'High Risk']
```

## 📞 Next Steps

1. Review high-risk customers in `results/high_risk_customers.csv`
2. Implement retention strategies for high-risk customers
3. Monitor model performance over time
4. Retrain model quarterly with new data
