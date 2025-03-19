###################################################################################################
#exclusion model (without top 14 predictors)
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

# 1) Load the data
data = pd.read_csv("processed_data.csv")

# 2) Define and transform the target
y = data["self_efficacy"]
y = y * 15 + 100

# List of 14 predictors to drop
features_to_drop = [
    "motivation_to_mastering",
    "meaning_in_life",
    "proud",
    "posotive_emotions",  
    "learning_goal",
    ".data_Malaysia",
    ".data_Japan",
    "student_cooperation",
    "scared",
    "parent_emotion_support",
    "reading_self_concept",
    "attitudes_competition",
    "fear_of_failure",
    "empathy"
]

# Drop target column from the dataset 
if "self_efficacy" in data.columns:
    data.drop("self_efficacy", axis=1, inplace=True)

# 3) Drop the 14 specified predictors
data.drop(columns=features_to_drop, inplace=True, errors='ignore')

# 4) Outer cross-validation
cv_outer = KFold(n_splits=4, shuffle=True, random_state=100)
outer_results = []

for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    
    # Inner CV for hyperparameter tuning
    cv_inner = KFold(n_splits=4, shuffle=True, random_state=100)
    
    # Define the base model
    model = XGBRegressor(random_state=100)
    
    # Define the parameter space
    space = {
        'max_depth': [5, 10, 15, 20, 25],
        'n_estimators': [50, 100, 150]
    }
    
    # GridSearchCV with multiple scoring metrics, refitting on "r2"
    search = GridSearchCV(
        estimator=model,
        param_grid=space,
        scoring=["r2", "neg_mean_squared_error", "neg_mean_absolute_error"],
        refit="r2",
        cv=cv_inner,
        return_train_score=True,
        n_jobs=6
    )
    
    # Fit the search
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Evaluate on the outer test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    outer_results.append({
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "best_params": search.best_params_
    })

# 5) Summarize outer-fold results
outer_results_df = pd.DataFrame(outer_results)
print("Outer CV results (per fold):")
print(outer_results_df)

mean_r2 = outer_results_df["r2"].mean()
std_r2  = outer_results_df["r2"].std()

outer_results_df.to_csv("xgb_exclusion_model.csv", index=False)
