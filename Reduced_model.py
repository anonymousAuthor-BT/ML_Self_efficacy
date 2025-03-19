
################################################################################################
#reduced model (with only the top 14 predictors)
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

# 1) Load the data
data = pd.read_csv("processed_data.csv")

# 2) Define the target
y = data["self_efficacy"]
y = y * 15 + 100

# 3) Define the subset of features 
features_to_keep = [
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

# Drop the old target column
if "self_efficacy" in data.columns:
    data.drop("self_efficacy", axis=1, inplace=True)

# 4) Create the feature matrix X from the specified columns
X = data[features_to_keep]

# 5) Outer cross-validation
cv_outer = KFold(n_splits=4, shuffle=True, random_state=100)
outer_results = []

for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
    
    # Inner CV for hyperparameter tuning
    cv_inner = KFold(n_splits=4, shuffle=True, random_state=100)
    
    # Define model
    model = XGBRegressor(random_state=100)
    
    space = {
        'max_depth': [5, 10, 15, 20, 25],
        'n_estimators': [50, 100, 150]
    }
    
    # Define the search
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
    result = search.fit(X_train, y_train)
    best_model = result.best_estimator_
    
    # Evaluate on the outer test set
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    outer_results.append({
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "best_params": result.best_params_
    })

# 6) Summarize outer-fold results
outer_results_df = pd.DataFrame(outer_results)
print("Outer CV results:")
print(outer_results_df)

mean_r2 = outer_results_df["r2"].mean()
std_r2  = outer_results_df["r2"].std()

outer_results_df.to_csv("xgb_reduced_model.csv", index=False)

