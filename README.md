# FairHiringBias-Mitigation-NU
This project investigates bias in machine learning hiring decisions and applies fairness-aware techniques to ensure equitable treatment across gender groups. Developed for the Nile University AI Fairness Challenge, the project explores how model performance and fairness can coexist in real-world decision systems.

This project aims to develop machine learning models that predict candidate hiring decisions based on multiple attributes while ensuring fairness across genders. I:

  - Trained several classifiers (Logistic Regression, Random Forest).
  - Evaluated fairness using Demographic Parity, Equal Opportunity, and Average Odds Difference.
  - Used SHAP and LIME for interpretability.
  - Applied Reweighing, Counterfactual Augmentation, and Feature Debiasing for fairness enhancement.


## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Data Preprocessing](#data_preprocessing)
- [Model Architecture and Performance](#model-architecture--performance)
  - [Model Architecture](#1.model-architecture)
  - [Model Performance](#2.model-performance)
- [Fairness Analysis](#fairness-analysis)
  - [Plots](#1-plots)
  - [Metrics](#2-metrics)
- [Explainability](#explainability)
- [Bias Mitigation Techniques](#bias-mitigation-techniques)

## Dataset Overview

The dataset contains **1,500 records with 11 columns**, representing various features relevant to hiring decisions. Each row corresponds to a unique job applicant, and the goal is to **predict whether the applicant was hired based on their attributes.**

- **Observations:**
  - No missing values in any column (the dataset is clean and complete).
  - About **50.8%** male and **49.2%** female data.
  - Data has no outliers.
  - The dataset includes a sensitive attribute `Gender`, which is the focus for fairness analysis.
  - Features are already numerically encoded, making them suitable for machine learning models.


## Data Preprocessing

To ensure consistent scaling and appropriate handling of categorical variables, several preprocessing steps were applied:
  - **Min-Max Scaling:**
    - Continuous numerical features such as `Age`, `EducationLevel`, `ExperienceYears`, `PreviousCompanies`, `DistanceFromCompany`, `InterviewScore`, `SkillScore`, and `PersonalityScore` were normalized using MinMaxScaler from sklearn.preprocessing. This transformation scales all values into the range `0`, `1` improving the performance and convergence behavior of many ML algorithms.
    
The fitted scaler was also saved using `joblib` for reproducibility and potential use during inference `(minmax_scaler.pkl)`.
```
joblib.dump(scaler, 'minmax_scaler.pkl')
```

  - **One-Hot Encoding:**
    - The categorical feature `RecruitmentStrategy` was transformed into binary indicator columns using one-hot encoding via pd.get_dummies().
    - This prevents the model from assuming any ordinal relationship between different strategies, which is crucial for unbiased learning.

These preprocessing steps resulted in a fully numerical and normalized dataset, ready for training fairness-aware machine learning models.


## Model Architecture and Performance

To begin modeling, the dataset was first split into features (X) and the target variable `HiringDecision`. 
- **A stratified train-test split** was then applied to ensure that the distribution of the hiring decision labels was preserved across both training and testing sets.
- This stratification is essential when dealing with **imbalanced classes** to ensure consistent evaluation.
```
# Sample 80% of males, 20% of females
biased_train = pd.concat([
    df_male.sample(frac=0.8, random_state=42),
    df_female.sample(frac=0.2, random_state=42)
])

# Shuffle the biased training set
biased_train = biased_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
```

### 1. Model Architecture

#### 1.1 Initial Model (Logistic Regression)

**Logistic Regression classifier** was trained using this biased dataset. 
  - The model was trained with a maximum of `1000 iterations` for convergence and a fixed random seed `random_state=42` for reproducibility.

#### 1.2 Best Model (Random Forest)

**Random Forest Classifier** was trained with carefully tuned hyperparameters. This ensemble learning method aggregates the predictions of multiple decision trees, which enhances model robustness and generalization.

**Model Hyperparameters:**
```
rf_clf = RandomForestClassifier(
    n_estimators=80,
    max_depth=20, 
    min_samples_split=7,
    min_samples_leaf=6, 
    max_features='sqrt',
    random_state=42
)
```
- This model achieved **the highest performance** before applying fairness mitigation techniques.

### 2. Model Performance

#### 2.1 Logistic Regression Classifier Results
- **Training Accuracy:**  `87.2%`
- **Test Accuracy:** `85.33%`

| Class / Metric   | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| **Class 0**      | 0.88      | 0.92   | 0.90     |
| **Class 1**      | 0.80      | 0.71   | 0.75     |
| **Macro Avg**    | 0.84      | 0.81   | 0.82     |
| **Weighted Avg** | 0.85      | 0.85   | 0.85     |
| **Accuracy**     | –         | –      | **0.85** |

#### 2.2 Random Forest Classifier Results
- **Training Accuracy:** ` 91.5%`
- **Test Accuracy:** `88.7%`

| Class / Metric   | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| **Class 0**      | 0.89      | 0.95   | 0.92     |
| **Class 1**      | 0.86      | 0.75   | 0.80     |
| **Macro Avg**    | 0.88      | 0.85   | 0.86     |
| **Weighted Avg** | 0.89      | 0.89   | 0.88     |
| **Accuracy**     | –         | –      | **0.8867** |

## Fairness Analysis

To assess the fairness of the model, **Demographic Parity**, **Equal Opportunity**, and **Average Odds Difference** were evaluated with respect to the `Gender` attribute.

### 1. Fairness Results:
- **Demographic Parity:** `-0.0919`
- **Equal Opportunity:** `-0.1802`
- **Average Odds Difference:** `-0.0855`

### 2. Plots

The plot shows the distribution of hiring predictions by `gender`:
<div align="center">
  <img src="plots/Gender_distribution.png" alt="SSVEP" width="2000"/>
</div>

| Gender     | Predicted Not Hired (0) | Predicted Hired (1) |
|------------|--------------------------|----------------------|
| Male (0)   | 112                      | 32                   |
| Female (1) | 107                      | 49                   |

### 3. Conclusion
- **Females were hired 9.2% more often than males overall.**
- Qualified females (y_true=1) had **18%** higher recall than qualified males.
- On average, **the model favors females by 8.6%.**


