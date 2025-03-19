The original student questionnaire data and school principal data can be sourced from the PISA website: https://www.oecd.org/en/data/datasets/pisa-2018-database.html.

The R scripts "data_preprocess_school_data" and "data_preprocess_student_data" can be used for data preprocessing. When combined, these scripts generate the CSV file "processed_data".

The Python scripts "Lasso_algorithm" and "XGB_algorithm" can be used to train and evaluate the models, producing the following model outputs: "statistics_lasso.csv", "statistics_XGB.csv", "resultslasso_inner", "resultsxgb_inner", and "XGB_feature_importance.xlsx".

The Python scripts "Reduced_model" and "Exclusion_model" were used to build additional predictive models for comparison with the full model, resulting in the output files: "xgb_exclusion_model" and "xgb_reduced_model".
