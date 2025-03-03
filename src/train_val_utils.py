import logging
import numpy as np
import pandas as pd
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import os
import json, glob
import joblib
from io import StringIO

#Helper functions
def weighted_soft_voting(probs_list, weights):
    weighted_sum = sum(w * p for w, p in zip(weights, probs_list))
    return weighted_sum / sum(weights)  # Normalize probabilities
    
def get_best_voting_weights(path):
    with open(path, "r") as f:
        data = json.load(f)

    # If JSON is a list of dicts, find the one with the highest F1-score
    if isinstance(data, list):
        return max(data, key=lambda x: x.get("f1_score", 0))

    return data  # If it's already a dict, return as is

def get_best_params(json_file_path):
    """
    Reads a JSON file containing multiple dictionaries, extracts the macro F1-score 
    from each, and returns the dictionary with the highest macro F1-score.
    
    :param json_file_path: Path to the JSON file.
    :return: Dictionary with the highest macro F1-score.
    """
    # Load JSON file
    with open(json_file_path, "r", encoding="utf-8") as file:
        data_list = json.load(file)  # Assuming it's a list of dictionaries

    # Function to extract macro F1-score
    def extract_macro_f1(classification_report):
        try:
            # Convert the classification report into a structured format
            lines = classification_report.strip().split("\n")
            for line in lines:
                if "macro avg" in line:
                    parts = line.split()
                    return float(parts[-2])  # Extract F1-score (2nd last value)
        except Exception as e:
            print(f"Error processing classification report: {e}")
        return -1  # Default value if parsing fails

    # Find the dictionary with the highest macro F1-score
    best_entry = max(data_list, key=lambda d: extract_macro_f1(d["classification_report"]))
    return best_entry



#fine tunning 
def finetune_logistic_regression(X_train, y_train, X_val, y_val, metadata_path="Models/logistic_regression_finetune.json"):
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    class_weights = [{0: w0, 1: w1} for w0, w1 in product([0.5, 1, 2, 3, 4], [3, 5, 7, 9, 12])]

    all_models_metadata = []


    for weight in class_weights:
        
        model_lr = LogisticRegression(class_weight=weight, random_state=42, max_iter=1000)
        model_lr.fit(X_train, y_train)

        # Predict probabilities
        y_scores = model_lr.predict_proba(X_val)[:, 1]

        
        # Optimize decision threshold
        thresholds = np.linspace(0.1, 0.9, 100)
        best_threshold, best_threshold_score = 0.5, 0

        for threshold in thresholds:
            y_pred_custom = (y_scores > threshold).astype(int)
            macro_f1 = f1_score(y_val, y_pred_custom, average='macro')
            if macro_f1 > best_threshold_score:
                best_threshold_score = macro_f1
                best_threshold = threshold


        # Store model metadata
        model_metadata = {
            'class_weight': weight,
            'threshold': best_threshold,
            'macro_f1': f1_score(y_val, y_pred_custom, average='macro'),
            'auc': roc_auc_score(y_val, y_scores),
            'accuracy': accuracy_score(y_val, y_pred_custom),
            'precision': precision_score(y_val, y_pred_custom),
            'recall': recall_score(y_val, y_pred_custom),
            'confusion_matrix': confusion_matrix(y_val, y_pred_custom).tolist()
    
        }
        all_models_metadata.append(model_metadata)

       
    # Save all models' metadata to JSON
    with open(metadata_path, "w") as f:
        json.dump(all_models_metadata, f, indent=4)


def finetune_random_forest(X_train, y_train, X_val, y_val, metadata_path="Models/random_forest_finetune.json"):

    # Ensure metadata directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Define hyperparameter search space
    max_depth_values = range(3, 11)  # From 3 to 10
    n_estimators_values = range(5, 51, 5)  # From 5 to 50 in steps of 5

    all_models_metadata = []  # Store metadata for all models
    
    # Iterate over hyperparameter combinations
    for max_depth, n_estimators in product(max_depth_values, n_estimators_values):

        # Train Random Forest Classifier
        model_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model_rf.fit(X_train, y_train)

        # Get Predictions
        y_pred_rf = model_rf.predict(X_val)
        y_scores_rf = model_rf.predict_proba(X_val)[:, 1]  # Get probability scores

        # Evaluate Model
        macro_f1 = f1_score(y_val, y_pred_rf, average='macro')
        auc_score = roc_auc_score(y_val, y_scores_rf)  # Compute AUC score

        # Store model metadata
        model_metadata = {
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'macro_f1': macro_f1,
            'auc': auc_score
        }
        all_models_metadata.append(model_metadata)
    # Save all models' metadata to JSON file
    with open(metadata_path, "w") as f:
        json.dump(all_models_metadata, f, indent=4)


def finetune_mlp_classifier(X_train, y_train, X_val, y_val, metadata_path="Models/mlp_finetune.json"):
    # Ensure metadata directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    # Apply SMOTE to balance the training set
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Define hyperparameter search space
    hidden_layer_sizes_list = [(64, 32), (128, 64), (128, 64, 32)]
    alpha_values = [0.0001, 0.001, 0.01, 0.1]

    best_models = []
    
    # Iterate over all hyperparameter combinations
    for hidden_layers, alpha in product(hidden_layer_sizes_list, alpha_values):

        model_nn = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=alpha,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )

        model_nn.fit(X_train_resampled, y_train_resampled)
        y_scores = model_nn.predict_proba(X_val)[:, 1]  # Get probability scores

        # Compute AUC score

        # Optimize decision threshold
        thresholds = np.linspace(0.1, 0.9, 100)
        best_threshold, best_threshold_score = 0.5, 0

        for threshold in thresholds:
            y_pred_custom = (y_scores > threshold).astype(int)
            auc_score = roc_auc_score(y_val, y_scores)
            macro_f1 = f1_score(y_val, y_pred_custom, average='macro') 

            if macro_f1 > best_threshold_score:
                best_threshold_score = macro_f1
                best_threshold = threshold


        cm = confusion_matrix(y_val, (y_scores > best_threshold).astype(int))
        report = classification_report(y_val, (y_scores > best_threshold).astype(int), digits=4)

        model_info = {
            'hidden_layers': hidden_layers,
            'alpha': alpha,
            'threshold': best_threshold,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'auc' : auc_score,
        }
        best_models.append(model_info)

    # Save all models to JSON
    with open(metadata_path, "w") as f:
        json.dump(best_models, f, indent=4)



#training best model
def best_logistic_regression(X_train, y_train, weights):
    # Train Logistic Regression Model with Class Weight
    lr_model = LogisticRegression(class_weight=weights, random_state=42, max_iter=1000)  # Adjust weight to reduce FN
    lr_model.fit(X_train, y_train)
    return lr_model


def best_random_forest(X_train, y_train, hyperparams):
    # Prepare data
    
    # Train Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=hyperparams['n_estimators'], max_depth=hyperparams['max_depth'], random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model


def best_mlp_classifier(X_train, y_train, hyperparams):
    # Train MLP Model
    mlp_model = MLPClassifier(
            hidden_layer_sizes=hyperparams['hidden_layers'],
            activation='relu',
            solver='adam',
            alpha=hyperparams['alpha'],
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
    mlp_model.fit(X_train, y_train)

    return mlp_model
    
    
    
#fine tunning voting weights
def finetune_voting_weights(adjusted_probs_poly, adjusted_probs_rf, adjusted_probs_nn, y_val):
    # Convert y_val to 1D if one-hot encoded
    y_val_1d = np.argmax(y_val, axis=1) if len(y_val.shape) > 1 and y_val.shape[1] > 1 else y_val

    # Define weight combinations
    weight_options = [
    (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2),
    (3, 1, 1), (1, 3, 1), (1, 1, 3), (2, 2, 1),
    (2, 1, 2), (1, 2, 2), (3, 2, 1), (2, 3, 1),
    (1.5, 1, 1), (1, 1.5, 1), (1, 1, 1.5), (2.5, 1, 1),
    (1, 2.5, 1), (1, 1, 2.5), (3, 1.5, 1), (1.5, 3, 1),
    (2.5, 3, 1), (3, 2.5, 1), (4, 3, 2), (3, 4, 2),
    (0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 0, 1), (2, 1, 0), (0, 2, 1),
    (0, 1, 2), (1, 0, 2), (3, 0, 1), (3, 1, 0), (0, 3, 1), (0, 1, 3),
    (1, 3, 0), (1, 0, 3), (1.5, 0, 1), (1, 0, 1.5), (1, 1.5, 0), (1.5, 1, 0),
    (2.5, 0, 1), (2.5, 1, 0), (1, 2.5, 0), (1.5, 0, 2), (0, 1.5, 2),
    (1, 1, 0), (0, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)
]


    results = []

    for weights in weight_options:
        # Perform weighted soft voting
        final_probs = weighted_soft_voting([adjusted_probs_poly, adjusted_probs_rf, adjusted_probs_nn], weights)
        y_pred = np.argmax(final_probs, axis=1)  # Convert probabilities to class labels

        auc = roc_auc_score(y_val_1d, y_pred, average="macro")
        f1 = f1_score(y_val_1d, y_pred, average="macro")
        cm = confusion_matrix(y_val_1d, y_pred)
        report = classification_report(y_val_1d, y_pred, digits=4)
        results.append({"weights": weights, "auc": auc, "confusion_matrix": cm.tolist(), "classification_report": report})
        logging.info(f"Weights: {weights} - Macro F1: {f1:.4f}")

    # Save results to JSON
    results_path = "Models/ensemble_weight.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)




#testing functions
def save_model(lr_model, rf_model, mlp_model, scaler, poly, params, file_prefix='Final Model/'):
  
  joblib.dump(lr_model, f'{file_prefix}poly.pkl')
  joblib.dump(rf_model, f'{file_prefix}rf.pkl')
  joblib.dump(mlp_model, f'{file_prefix}nn.pkl')
  joblib.dump(scaler, f'{file_prefix}scaler.pkl')
  joblib.dump(poly, f'{file_prefix}poly_features.pkl')
  with open(f'{file_prefix}best_wieghts.json', "w") as f:
        json.dump(params, f, indent=4)
  
def load_saved_model(file_prefix='Final Model/'):
    # Load the saved models and preprocessing objects
    lr_model = joblib.load(f"{file_prefix}poly.pkl")
    rf_model = joblib.load(f"{file_prefix}rf.pkl")
    mlp_model = joblib.load(f"{file_prefix}nn.pkl")
    scaler = joblib.load(f"{file_prefix}scaler.pkl")
    poly = joblib.load(f"{file_prefix}poly_features.pkl")
    params = get_best_voting_weights(f"{file_prefix}best_wieghts.json")

    return scaler, poly, lr_model, rf_model, mlp_model, params

def test_model(df_test, scaler, poly, lr_model, rf_model, mlp_model, params):
# Preprocess the test data
    X_test = df_test.drop(columns=['Class'])  # Assuming 'Class' is the target column
    X_test_scaled = scaler.transform(X_test)
    X_test_poly = poly.transform(X_test_scaled)
    y_test = df_test['Class']
    # Get prediction probabilities from each model
    probs_lr = lr_model.predict_proba(X_test_poly)
    probs_rf = rf_model.predict_proba(X_test_poly)
    probs_mlp = mlp_model.predict_proba(X_test_scaled)
    
    lr = (probs_lr > 0.5364).astype(int)
    rf = (probs_rf > 0.5).astype(int)
    mlp = (probs_mlp > 0.9).astype(int)

    
    # Ensemble predictions using weighted soft voting
    final_probs = weighted_soft_voting([lr, rf, mlp], params['weights'])
    predictions = np.argmax(final_probs, axis=1)

    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    auc_score = roc_auc_score(y_test, final_probs[:, 1])  # Assuming binary classification

    return conf_matrix, report, auc_score, predictions
