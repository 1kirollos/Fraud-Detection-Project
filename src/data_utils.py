import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler



# Function to load data
def load_data(path):
    try:
        train_df = pd.read_csv(path[0])
        val_df = pd.read_csv(path[1])
        return train_df, val_df
    except Exception as e:
        raise

def load_test(path):
    try:
        test_df = pd.read_csv(path)
        return test_df
    except Exception as e:
        raise
# Helper function to create new features
def generate_combined_features(df, col1, col2):
    try:
        transformations = {
            f"{col1}_{col2}_sum": df[col1] + df[col2],
            f"{col1}_{col2}_diff": df[col1] - df[col2],
            f"{col1}_{col2}_abs_diff": abs(df[col1] - df[col2]),
            f"{col1}_{col2}_product": df[col1] * df[col2],
            f"{col1}_{col2}_ratio": df[col1] / (df[col2] + 1e-6),  # Avoid div by zero
            f"{col1}_{col2}_mean": (df[col1] + df[col2]) / 2,
            f"{col1}_{col2}_log_prod": np.log1p(abs(df[col1] * df[col2])),
        }

        # Convert dictionary to DataFrame and concatenate with the original DataFrame
        new_features_df = pd.DataFrame(transformations)
        df = pd.concat([df, new_features_df], axis=1)
        return df
    except Exception as e:
        raise

def generate_all_combinations(df, feature_combinations):
    try:
        for feature, feature_list in feature_combinations:
            for col in feature_list:
                df = generate_combined_features(df, feature, col)
        return df
    except Exception as e:
        raise

def filter_features(df, target, threshold):
    try:
        correlation_matrix = df.corr()
        relevant_corr = correlation_matrix[[target]]
        filtered_corr = relevant_corr[abs(relevant_corr[target]) > threshold]
        return df[filtered_corr.index]
    except Exception as e:
        raise

def split_data(df_train, df_val, target):
    try:
        X_train = df_train.drop(columns=[target])
        y_train = df_train[target]
        X_val = df_val.drop(columns=[target])
        y_val = df_val[target]
        return X_train, y_train, X_val, y_val
    except Exception as e:
        raise

def scaler(X_train, X_val):
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        return X_train_scaled, X_val_scaled, scaler
    except Exception as e:
        raise
    
def poly(X_train, X_val, degree = 2, interactions = False):
    try:
        poly = PolynomialFeatures(degree=degree, interaction_only=interactions)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        return X_train_poly, X_val_poly, poly
    except Exception as e:
        raise
      
def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    correct_order = ['Class', 'V11', 'V4', 'Time_V11_product', 'Time_V4_product', 'V2', 'V9',
                     'V5', 'V1', 'V18', 'Time_V3_product', 'V7', 'V3', 'V16',
                     'Time_V12_product', 'V10', 'V12', 'V14', 'V17']
    
    return df[correct_order]


def preparing_test(test_file_path):
    
    test_df = load_test(test_file_path)
    
    # Define feature combinations
    amount_features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V10', 'V20', 'V23']
    time_features = ['V25', 'V22', 'V15', 'V12', 'V11', 'V5', 'V4', 'V3', 'V1']
    feature_combinations = [('Amount', amount_features), ('Time', time_features)]

    # Generate feature combinations
    test_df_enhanced = generate_all_combinations(test_df, feature_combinations)

    # Filter features
    test_df_final = filter_features(test_df_enhanced, 'Class', 0.07)
    
    #ordering columns
    df_test_ordered = reorder_columns(test_df_final)
    return df_test_ordered