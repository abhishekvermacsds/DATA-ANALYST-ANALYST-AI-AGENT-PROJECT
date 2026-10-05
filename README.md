# IPL 2026 Match Winner Prediction

A machine learning project to predict IPL match winners using ball-by-ball data from 2008–2025.

## Models
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Neural Network
- Stacking Ensemble

## Dataset
IPL Ball-by-Ball Data (2008–2025)

## Target
Predict which team wins a given IPL match.

## Best Model
**Logistic Regression — 56.5% test accuracy**

## Main Steps
1. Load and preprocess IPL data
2. Create match-level features
3. Split data by season
4. Train and evaluate models
5. Predict IPL 2026 fixtures
6. Run playoff and Monte Carlo simulations

## Run
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost tensorflow matplotlib seaborn kagglehub
python IPL_2026_Match_Winner_Prediction.py
```

## File
`IPL_2026_Match_Winner_Prediction.py`

## Author
Abhishek Verma
