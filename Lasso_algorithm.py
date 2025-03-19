##############################################################################method 3: use for-loop
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Lasso
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

#define target
data = pd.read_csv("processed_data.csv")

#define target
y = data["self_efficacy"]
y = y * 15 + 100
data.drop("self_efficacy", axis=1, inplace = True)

#define features
X = data.iloc[:, 1:]

##############################################################################
# configure the cross-validation procedure
cv_outer = KFold(n_splits=4, shuffle=True, random_state=100)

# enumerate splits
outer_results = list()
inner_results = list()

r2_0 = 0

for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :] 
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix] 

    cv_inner = KFold(n_splits=4, shuffle=True, random_state=100) 
    model = Lasso() 

    space = dict() 
    space['alpha'] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1] 

    search = GridSearchCV(model, space, 
                          scoring = ["r2", "neg_mean_squared_error",
                                     "neg_mean_absolute_error"], 
                          refit= "r2", cv=cv_inner,
                          return_train_score=True,
                          n_jobs = 6
                          ) 

    results = search.fit(X_train, y_train) 
    best_model = results.best_estimator_ 

#get inner results:
    inner_results.append(results.cv_results_)

#outer evaluation and get the results
    yhat = best_model.predict(X_test) 
    rmse = math.sqrt(mean_squared_error(y_test, yhat))
    mae = mean_absolute_error(y_test, yhat) 
    r2 = r2_score(y_test, yhat)
    outer_results.append([rmse, mae, r2]) 
    
    if r2 > r2_0:
       r2_0 = r2
       importance = np.transpose(best_model.coef_)
       feat_importances = pd.DataFrame(importance, index = data.iloc[:, 1:].columns)

    print('>results=%.3f, best_score=%.3f, best_param=%s' % (r2, results.best_score_, results.best_params_))

results_lasso_outer = pd.DataFrame(outer_results)
results_lasso_inner = pd.DataFrame(inner_results)

#get the feature importance (not implemented due to the better-performance of XGBoost model)
#feat_importances.to_csv("lasso_feature_importance.csv")

mean_r2_test = list()
mean_r2_train = list()
mean_neg_MSE_train = list()
mean_neg_MSE_test = list()
mean_neg_mean_absolute_error_test = list()
mean_neg_mean_absolute_error_train = list()

for i in range(4):
    inner_res = pd.DataFrame(inner_results[i])
    max_r2_test = inner_res[["split0_test_r2", "split1_test_r2", "split2_test_r2", "split3_test_r2"]].max()
    mean_r2_test.append(np.mean(max_r2_test))
    
    max_r2_train = inner_res[["split0_train_r2", "split1_train_r2", "split2_train_r2", "split3_train_r2"]].max()
    mean_r2_train.append(np.mean(max_r2_train))
        
    max_neg_MSE_train = inner_res[["split0_train_neg_mean_squared_error", "split1_train_neg_mean_squared_error", "split2_train_neg_mean_squared_error", "split3_train_neg_mean_squared_error"]].max()
    mean_neg_MSE_train.append(np.mean(max_neg_MSE_train))

    max_neg_MSE_test = inner_res[["split0_test_neg_mean_squared_error", "split1_test_neg_mean_squared_error", "split2_test_neg_mean_squared_error", "split3_test_neg_mean_squared_error"]].max()
    mean_neg_MSE_test.append(np.mean(max_neg_MSE_test))
    
    max_neg_mean_absolute_error_train = inner_res[["split0_train_neg_mean_absolute_error", "split1_train_neg_mean_absolute_error", "split2_train_neg_mean_absolute_error", "split3_train_neg_mean_absolute_error"]].max()
    mean_neg_mean_absolute_error_train.append(np.mean(max_neg_mean_absolute_error_train))

    max_neg_mean_absolute_error_test = inner_res[["split0_test_neg_mean_absolute_error", "split1_test_neg_mean_absolute_error", "split2_test_neg_mean_absolute_error", "split3_test_neg_mean_absolute_error"]].max()
    mean_neg_mean_absolute_error_test.append(np.mean(max_neg_mean_absolute_error_test))

statistics = pd.DataFrame([np.mean(mean_r2_train), np.mean(mean_r2_test),
              abs(np.mean(mean_neg_mean_absolute_error_train)),
              abs(np.mean(mean_neg_mean_absolute_error_test)),
              np.sqrt(abs(abs(np.mean(mean_neg_MSE_train)))),
              np.sqrt(abs(abs(np.mean(mean_neg_MSE_test))))])
              
statistics.to_csv("statistics_lasso.csv")
results_lasso_inner.to_csv("resultslasso_inner.csv")