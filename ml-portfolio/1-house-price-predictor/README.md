# California House Price Predictor
End-to-end machine learning project: feature engineering, model comparison with cross-validation,
hyperparameter tuning, held-out evaluation, explainability and an interactive web app.

**Stack:** Python, scikit-learn, pandas, matplotlib, Streamlit

## Run
    pip install -r ../requirements.txt
    python train.py          # trains, tunes, prints results, saves model + plots
    streamlit run app.py     # interactive predictor

## Pipeline
1. Engineered features (bedroom ratio, people per room, income per room)
2. Compared Baseline, Ridge, Random Forest, Gradient Boosting (3-fold CV)
3. Tuned Gradient Boosting with RandomizedSearchCV
4. Evaluated on 20% held-out test set
5. Explained predictions with permutation feature importance

## Results (fill in after running train.py)
| Model | CV RMSE |
|---|---|
| Baseline | 1.156 +/- 0.007 |
| Ridge | 2.037 +/- 1.863 |
| Random Forest | 0.517 +/- 0.001 |
| Gradient Boosting (tuned) | 0.475 +/- 0.002 |

Test set: MAE = $29435.13436904227  |  R² = 0.8467610953695921
