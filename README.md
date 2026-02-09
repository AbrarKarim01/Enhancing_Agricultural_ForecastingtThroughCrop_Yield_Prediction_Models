# Enhancing Agricultural Forecasting Through Crop Yield Prediction Models

## Overview
This project explores global crop yield prediction using classical machine learning models. The workflow covers exploratory data analysis, preprocessing, feature engineering (PCA, t-SNE), and model evaluation across multiple regressors. The goal is to understand which features (temperature, rainfall, pesticide usage, location, crop type) best explain yield and which models perform most reliably.

## Dataset
Source: Kaggle - Crop Yield Prediction Dataset
https://www.kaggle.com/datasets/mrigaankjaswal/crop-yield-prediction-dataset/data

Key fields used in the notebook:
- Area (country/region)
- Item (crop type)
- Year
- avg_temp
- average_rain_fall_mm_per_year
- pesticides_tonnes
- hg/ha_yield (target)

The notebook loads a local CSV named `yield_df.csv`.

## Methodology
1. Exploratory Data Analysis
   - Crop distribution, top countries by yield, and yield trends over time
   - Relationship plots for yield vs temperature, rainfall, and pesticides
   - Correlation heatmap
2. Preprocessing
   - Drop unused columns (e.g., `Unnamed: 0`)
   - One-hot encode `Area` and `Item`
   - Scale features with MinMaxScaler
   - Train/test split
3. Modeling and Evaluation
   - Decision Tree Regressor
   - Random Forest Regressor
   - XGBoost Regressor
   - Linear Regression
   - Metrics: R2, MAE, MSE, RMSE
4. Validation and Feature Engineering
   - K-Fold evaluation for Random Forest
   - PCA and t-SNE dimensionality reduction
   - Compare R2 scores for Original vs PCA vs t-SNE
5. Result Analysis
   - Residual plot
   - Prediction vs actual scatter plot

## Results
Model performance on the test split:

| Model | R2 | MAE | MSE | RMSE |
| --- | --- | --- | --- | --- |
| Decision Tree | 0.9584 | 6047.40 | 302046195.80 | 17379.48 |
| Random Forest | 0.9734 | 5553.74 | 193267849.39 | 13902.08 |
| XGBoost | 0.9514 | 11726.99 | 352549664.00 | 18776.31 |
| Linear Regression | 0.7493 | 29811.75 | 1818523144.97 | 42644.15 |

Random Forest cross-validation (5-fold):
- Fold R2: 0.9749, 0.9719, 0.9743, 0.9767, 0.9701
- Average R2: 0.9736

Dimensionality reduction comparison (Random Forest):
- Original features R2: 0.9734
- PCA (2 components) R2: 0.7070
- t-SNE (2 components) R2: 0.9447

## Project Structure
- Project Code/Enhancing_Agricultural_Forecasting_through_Crop_Yield_Prediction_Models.ipynb
- Dataset/ (place `yield_df.csv` here or adjust the notebook path)
- Project Paper/
- Project Presentation/
- Project Proposal/

## How to Run
1. Ensure `yield_df.csv` is available (update the notebook path if needed).
2. Open the notebook:
   - Project Code/Enhancing_Agricultural_Forecasting_through_Crop_Yield_Prediction_Models.ipynb
3. Run all cells from top to bottom.

## Dependencies
Python libraries used:
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost

Install example:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Notes
- The notebook drops `Year` before training and uses one-hot encoded country and crop features.
- Update any file paths if your dataset is stored elsewhere.
