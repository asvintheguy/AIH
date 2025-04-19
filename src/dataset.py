"""
Dataset handling and analysis functions.
"""

import os
import pandas as pd
from src.kaggle_api import download_kaggle_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MAX_DATASET_SIZE_MB = float(os.getenv("MAX_DATASET_SIZE_MB", "300"))  # 300MB default limit
MAX_ROWS_SAMPLE = 50000  # For very large files, sample this many rows


def analyze_dataset(csv_path):
    """
    Analyze a CSV dataset and determine features and target.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        dict: Dictionary containing dataset analysis results
    """
    # Check file size
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"\n📄 CSV File Size: {file_size_mb:.2f} MB")
    
    # If file is too large, use sampling
    if file_size_mb > MAX_DATASET_SIZE_MB:
        raise ValueError(f"CSV file too large: {file_size_mb:.2f} MB (limit: {MAX_DATASET_SIZE_MB} MB)")
    elif file_size_mb > 100:  # For moderately large files (>100MB), use sampling
        print(f"⚠️ Large CSV file detected. Sampling {MAX_ROWS_SAMPLE} rows...")
        # Sample the data
        df = pd.read_csv(csv_path, nrows=MAX_ROWS_SAMPLE)
        print(f"   Sampled {len(df)} rows for analysis")
    else:
        # For smaller files, load the entire dataset
        df = pd.read_csv(csv_path)

    print(f"📄 Dataset Shape: {df.shape}")

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
    }


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