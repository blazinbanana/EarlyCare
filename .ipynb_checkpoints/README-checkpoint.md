# Advanced Liver Disease Detection for Early Palliative Care

## Overview
This project develops a machine learning model (Random Forest Classifier) to identify patients with advanced liver disease who would benefit from early palliative care interventions. The model analyzes routine blood test results to predict disease severity, addressing the critical gap in tools for timely palliative care referral.

**Key Achievement:** The final model achieved 99% accuracy in classifying advanced liver disease stages using only blood test parameters.

## Business & Clinical Problem
- **Problem:** Lack of systematic tools to identify patients needing early palliative care in liver disease
- **Impact:** Late palliative referrals lead to reduced quality of life, increased hospitalizations, and higher healthcare costs
- **Solution:** A predictive model that uses readily available blood test data to flag high-risk patients

## Data Source
- **Dataset:** [https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset?resource=download&select=Liver+Patient+Dataset+%28LPD%29_train.csv)
  
- **Features:**
Age of the patient                        
Gender of the patient                   
Total Bilirubin                         
Direct Bilirubin                        
Alkphos Alkaline Phosphotase            
Sgpt Alamine Aminotransferase           
Sgot Aspartate Aminotransferase         
Total Protiens                          
ALB Albumin                             
A/G Ratio Albumin and Globulin Ratio

- **Target:**
Result

## Methodology

### Data Preprocessing

- Handled missing values using:
SimpleImputer(strategy='median') for numerical columns
SimpleImputer(strategy='most_frequent') for categorical columns

- Scaled numerical features using StandardScaler to normalize feature magnitudes before model training.

- Encoded categorical features with OneHotEncoder(handle_unknown='ignore') to handle string-based variables.

- Addressed class imbalance using RandomOverSampler from the imblearn library, ensuring equal representation of both classes.

### Exploratory Data Analysis
![Feature Importance](assets/feature_importances.PNG)
![Data Distribution](assets/data_distribution_gender.PNG)
![Data Distribution](assets/data_distribution_age.PNG)

### Modeling Approach
Algorithms tested: Random Forest Classifier (primary), with potential extensions to Logistic Regression or Gradient Boosted Trees.

- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC were used to assess predictive performance.

- Final model: Random Forest Classifier

- Tuned via GridSearchCV with 5-fold cross-validation

Parameters optimized included:
max_depth
n_estimators
min_samples_split
max_features

## Results
| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Final Model | 99% | 99% | 99% | 99% |

**Key Insight:** The model successfully identifies [99]% of advanced liver disease cases, enabling earlier palliative care referrals.

## Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://earlycare-gkszkb6h3xpfyo9jnqdbp3.streamlit.app/)


Try the interactive demo where you can input blood test values and get predictions.

You can also input a batch of patient's records (for clinicians) as long as the column names are ordered well (exact matches) . The predictions will be saved as a csv file, with 1 indicating high risk and 2 indicating low risk.

## Technologies Used
- Python, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn for visualization
- Pickle for model serialization
- Streamlit for deployment

## Installation & Usage

### Running the Notebook
```bash
git clone https://github.com/blazinbanana/EarlyCare.git
cd EarlyCare
pip install -r requirements.txt
jupyter notebook notebooks/Advanced liver sickness detector.ipynb

