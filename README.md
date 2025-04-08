# Airline Churn Prediction – Comparative Machine Learning Models
This project focuses on identifying customers at risk of churn in the airline industry using a real-world dataset from Kaggle. The goal was to compare different machine learning models and select the best one for predicting disloyal customers, enabling more effective retention strategies.

## Objectives
- Predict customer churn based on satisfaction, flight experience, and demographics
- Compare multiple machine learning models using key performance metrics
- Optimize classification thresholds to maximize profit and retention effectiveness

## Dataset
- **Source**: Kaggle – Invistico Airline Dataset
- **Features**: Service ratings, travel class, demographics, delays, satisfaction, etc.
- **Target**: `Churn` (1 = disloyal, 0 = loyal)

## Models Tested
- Logistic Regression (GLM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Lasso Regression
- Naïve Bayes
- Partial Least Squares (PLS)
- Random Forest 🌟 (best)
- Gradient Boosting Machine (GBM)
- Neural Networks

## Preprocessing
- Missing value imputation using **MICE (pmm)**
- Transformation of service rating scales into ordinal categories
- Encoding categorical variables as factors
- Feature selection via correlation analysis and variance filtering
- Dataset split: training (75%), test (25%), scoring (10%)

## Evaluation & Results
Models were evaluated using:
- **Accuracy**
- **Sensitivity**
- **Specificity**
- **ROC-AUC**
- **Profit-based threshold tuning**

| Model              | ROC-AUC | Accuracy |
|-------------------|---------|----------|
| Random Forest      | 0.97    | 96.59%   |
| Neural Networks    | 0.96    | 96.16%   |
| Gradient Boosting  | 0.95    | 95.19%   |
| KNN                | 0.96    | 95.83%   |
| Logistic Regression| 0.94    | 90.15%   |

- **Best model**: Random Forest (high AUC, strong interpretability, good generalization)
- Lift Chart and Gain Curve analysis confirmed its high precision in identifying churners.

## Threshold Optimization
- Custom profit function: identifying a churner is 4x more valuable than misclassifying a loyal customer.
- Optimal classification threshold: **0.31**, maximizing expected profit while maintaining balance between sensitivity and specificity.

## Tools Used
- **R**, **RStudio**
- Packages: `caret`, `mice`, `randomForest`, `gbm`, `nnet`, `glmnet`, `Boruta`, `pROC`, etc.

*Developed as part of a Machine Learning course project (A.Y. 2024/2025), University of Milano-Bicocca.*
