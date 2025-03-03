import logging
from data_utils import load_data, load_test, generate_all_combinations, filter_features, split_data, scaler, poly, reorder_columns,preparing_test
from train_val_utils import *
import logging
import pprint

def train():                  
    #Load data
    train_df, val_df = load_data(["Data/train.csv", "Data/val.csv"])

    # Define feature combinations
    amount_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V10', 'V20', 'V23']
    time_features = ['V25', 'V22', 'V15', 'V12', 'V11', 'V5', 'V4', 'V3', 'V1']
    feature_combinations = [('Amount', amount_features), ('Time', time_features)]

    # Generate feature combinations
    train_df_enhanced = generate_all_combinations(train_df, feature_combinations)
    val_df_enhanced = generate_all_combinations(val_df, feature_combinations)

    # Filter features
    train_df_final = filter_features(train_df_enhanced, 'Class', 0.07)
    val_df_final = filter_features(val_df_enhanced, 'Class', 0.07)

    #sorting columns to match the order of the columns in the model
    ordered_val_df = reorder_columns(val_df_final)

    # Split data
    X_train, y_train, X_val, y_val = split_data(train_df_final, ordered_val_df, 'Class')

    # Scale data
    X_train_scaled, X_val_scaled, scaler_trans = scaler(X_train, X_val)

    # Polynomial features
    X_train_poly, X_val_poly, poly_trans = poly(X_train_scaled, X_val_scaled, degree=2, interactions=False)

    #fine tunning models
    finetune_logistic_regression(X_train_poly, y_train, X_val_poly, y_val)
    finetune_random_forest(X_train_poly, y_train, X_val_poly, y_val)
    finetune_mlp_classifier(X_train_scaled, y_train, X_val_scaled, y_val)

    #getting best parameters
    lr_params = get_best_params("Models metadata/lr_fineutne.json")
    rf_params = get_best_params("Models metadata/rf_fineutne.json")
    mlp_params = get_best_params("Models metadata/mlp_fineutne.json")
    lr_params['class_weight'] = {int(k): v for k, v in lr_params['class_weight'].items()}
    rf_params['class_weight'] = {int(k): v for k, v in rf_params['class_weight'].items()}
    mlp_params['class_weight'] = {int(k): v for k, v in mlp_params['class_weight'].items()}


    #training best models
    lr_model = best_logistic_regression(X_train_poly, y_train, lr_params['class_wights'])
    rf_model = best_random_forest(X_train_poly, y_train, rf_params)
    mlp_model = best_mlp_classifier(X_train_scaled, y_train, mlp_params)

    #models socres
    lr_scores = lr_model.predict_porba(X_val_poly)
    rf_scores = rf_model.predict_porba(X_val_poly)
    mlp_scores = mlp_model.predict_porba(X_val_scaled)

    #models predictions
    lr = (lr_scores > lr_params['best_threshold']).astype(int)
    rf = (rf_scores > 0.5).astype(int)
    mlp = (mlp_scores > mlp_params['best_threshold']).astype(int)


    #fine tunning best voting weights
    finetune_voting_weights(lr_model, rf_model, mlp_model, y_val)

    params = get_best_voting_weights("Models metadata/voting_fineutne.json")

    save_model(lr_model, rf_model, mlp_model, params, "Models metadata/voting_weights.json")

def test(test_file_path):
    #loading daata
    
    test_df = preparing_test(test_file_path)
    #loading models
    scaler_trans, poly_trans, lr_model, fr_model, mlp_model, params = load_saved_model()
    
    #testing models
    conf_matrix, report, auc_score, predictions = test_model(test_df, scaler_trans, poly_trans, lr_model, fr_model, mlp_model, params)
    
    #printing results
    print(conf_matrix)
    print(report)
    print(auc_score)
    return predictions
    
preds = test("Data/test.csv")