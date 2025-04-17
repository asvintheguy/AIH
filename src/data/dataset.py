"""
Dataset handling and analysis functions.
"""

import pandas as pd
from src.api.kaggle_api import download_kaggle_dataset


def analyze_dataset(csv_path):
    """
    Analyze a CSV dataset and determine features and target.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        dict: Dictionary containing dataset analysis results
    """
    df = pd.read_csv(csv_path)

    print(f"\n📄 Dataset Shape: {df.shape}")

    # Guess target column (last column)
    target = 'Outcome' if 'Outcome' in df.columns else df.columns[-1]

    # Separate features
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Remove target from features
    if target in numerical:
        numerical.remove(target)
    if target in categorical:
        categorical.remove(target)

    print(f"\n🔢 Numerical Features: {numerical}")
    print(f"🔤 Categorical Features: {categorical}")
    print(f"🎯 Target Column: {target}")

    return {
        "numerical_features": numerical,
        "categorical_features": categorical,
        "target_column": target,
        "data": df,
        "disease_name": "Diabetes"  # Default disease name
    }


def load_dataset(kaggle_url):
    """
    Load a dataset from Kaggle URL.
    
    Args:
        kaggle_url (str): URL of Kaggle dataset
        
    Returns:
        dict: Dictionary with dataset information and analysis
    """
    csv_file_path, readme = download_kaggle_dataset(kaggle_url)
    result = analyze_dataset(csv_file_path)
    result["readme"] = readme
    print(readme)
    return result


def infer_column_descriptions(df):
    """
    Infer and generate descriptions for dataset columns.
    
    Args:
        df (DataFrame): Pandas DataFrame
        
    Returns:
        dict: Dictionary mapping column names to descriptions
    """
    descriptions = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            min_val = df[col].min()
            max_val = df[col].max()
            unique_vals = df[col].nunique()
            descriptions[col] = f"Numeric | Range: [{min_val}, {max_val}] | Unique Values: {unique_vals}"
        else:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 10:
                descriptions[col] = f"Categorical | Values: {list(unique_vals)}"
            else:
                descriptions[col] = f"Categorical | Unique count: {len(unique_vals)}"
    return descriptions


def get_target_labels_mapping(dataset):
    """
    Infers what the 'high risk' and 'low risk' labels are based on the target column.
    
    Args:
        dataset (DataFrame): Dataset with target column
        
    Returns:
        tuple: Dictionary mapping target values to risk levels, and target column name
    """
    target_col = dataset.columns[-1]  # assuming the last column is the target
    unique_vals = dataset[target_col].dropna().unique()

    # Normalize for comparison
    normalized = [str(val).strip().lower() for val in unique_vals]

    # Heuristics to determine high risk label
    if "high" in normalized and "low" in normalized:
        return {"high": "High Risk", "low": "Low Risk"}, target_col
    elif set(normalized) == {"1", "0"} or set(normalized) == {"0", "1"}:
        return {"1": "High Risk", "0": "Low Risk"}, target_col
    else:
        print(f"⚠️ Unknown label format: {unique_vals}. Treating as binary with 1 = High Risk.")
        return {"1": "High Risk", "0": "Low Risk"}, target_col 