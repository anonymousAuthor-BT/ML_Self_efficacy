The XGBoost and Lasso algorithms were separately trained using the processed_data. Python code for model training and evaluation included "Lasso_algorithm.py" and "XGBoost_algorithm.py".


The original PISA data can be sourced from: https://www.oecd.org/en/data/datasets/pisa-2018-database.html

The R code "preprocess_data" can be used for data preprocessing, which leads to the csv. file "processed_data".

Python codes "Lasso_algorithm" and "XGB_algorithm" can be used to train and evaluate the models, which results in the model outputs: "statistics_lasso.csv", "statistics_XGB.csv", "resultslasso_inner", "resultsxgb_inner", and "XGB_feature_importance.xlxs".

Python codes "Reduced_model" and "Exclusion_model" were used to build additional predictive models for comparisons with the full model, which led to output files: "xgb_exclusion_model" and "xgb_reduced_model".
