# Airline Satisfaction Prediction

Solution for the Kaggle [Playground Series S6E10: Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10).

## Problem
Binary classification: predict whether an airline passenger is satisfied, using flight details, passenger info and 13 service ratings. Metric: **ROC AUC** on predicted probabilities. 

## Data
- `train.csv`: 699,635 rows, 23 columns
- `test.csv`: 299,844 rows
- Target: `satisfaction` (44% satisfied, 56% not satisfied)
- Only `Arrival Delay in Minutes` has missing values (292) 

Data is synthetic and generated from a real airline passenger satisfaction dataset. It is not included in this repo (see Kaggle competition page).

## Approach
1. Clean column names, convert categorical columns to `category` dtype  
2. LightGBM baseline with early stopping
3. 5-fold Stratified CV, test predictions averaged across folds
4. Feature engineering: total delay, rating mean/sum/min/max/std, count of low and high ratings
5. (Planned) CatBoost + XGBoost, ensemble, original dataset as extra data

## Results

| Version | Description | CV AUC | Public LB |
|---|---|---|---|
| v1 | LightGBM, single split | 0.9578 (val) | 0.95804 |
| v2 | LightGBM, 5-fold | 0.95884 | 0.95831 |
| v3 | + feature engineering | TBD | TBD |
| v4 | CatBoost + XGBoost ensemble | TBD | TBD |

## Key learnings
- CV and public LB were close, so the validation setup is reliable
- Single split vs 5-fold: small but consistent gain from fold averaging

## Tech stack
Python, pandas, NumPy, scikit-learn, LightGBM

## How to run
1. Download data from the competition page into a `data/` folder
2. Install requirements: `pip install -r requirements.txt`
3. Open the notebooks in order from `notebooks/`

## Author
Sakshi Chaudhary
