# Titanic Survival Prediction

This project applies machine learning techniques to predict passenger survival outcomes from the 1912 Titanic disaster. The analysis is conducted as part of the [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition.

## 📊 Problem Statement

**Research Question**:  
Can we predict which passengers survived the Titanic disaster using available information such as age, gender, ticket class, and family relations? Understanding these patterns provides insights into survival factors and enables the development of predictive models that can be applied to similar scenarios.

## 🔍 Dataset

The Titanic dataset includes:
- Passenger demographics (age, gender, family)
- Ticket and fare details
- Cabin class and embarkation location
- Survival outcome (0 = No, 1 = Yes, for training set)

Datasets:
- `train.csv` (labeled data)
- `test.csv` (unlabeled data for Kaggle submission)

## 🧰 Methods and Workflow

### 1️⃣ Data Exploration & Preprocessing
- **Exploratory Data Analysis (EDA)**: 
  - Visualizations of age, fare distributions, survival rates by gender and class.
  - Correlation analysis to identify key predictors (e.g., Sex, Pclass, Fare).
- **Data Cleaning**:
  - Imputed missing values (`Age`, `Fare`, `Embarked`) with logical defaults (e.g., median or mode).
  - Removed outliers using the 99th percentile for `Fare` and `Age`.
- **Feature Engineering**:
  - Created `FamilySize`, `IsAlone` features.
  - Extracted and grouped `Title` from passenger names (e.g., Mr, Miss, Mrs).
  - One-hot encoded categorical variables.
- **Scaling**:
  - Applied for regression models; skipped for tree-based models.

### 2️⃣ Model Development

#### Regression Models:
- Logistic Regression
- Linear Discriminant Analysis (LDA)
- K-Nearest Neighbors (KNN, with Grid Search)

#### Tree-Based Models:
- Random Forest (with hyperparameter tuning)
- Extra Trees (with hyperparameter tuning)
- Gradient Boosted Trees (with hyperparameter tuning)

### 3️⃣ Hyperparameter Tuning
- Tuned `n_estimators`, `max_features`, `max_depth`, and `criterion` for tree-based models.
- Grid Search used for `KNN` hyperparameters (`k`, distance metrics, weights).

### 4️⃣ Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- ROC Curves and AUC
- Precision-Recall Curves
- Confusion Matrices


## 📈 Key Results

| Model                 | Accuracy  | AUC     | Best For                          |
|-----------------------|-----------|---------|----------------------------------|
| Logistic Regression   | 81%       | 0.85    | Interpretability                  |
| KNN (k=13)            | 81%       | ~0.83   | Distance-based learning           |
| Random Forest         | 82.3%     | 0.85    | Overall performance               |
| Gradient Boosting     | 81.1%     | 0.87    | Best AUC (classification confidence) |
| Extra Trees           | 81.7%     | 0.84    | Alternative tree-based model      |

**Insights**:
- Gender (`Sex`), title (`Mr`, `Mrs`, `Miss`), passenger class (`Pclass`), and fare were the strongest survival predictors.
- Social status, gender, and economic status significantly influenced survival probabilities.
