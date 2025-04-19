"""
Machine learning model training and evaluation.
"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# Load environment variables
load_dotenv()
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


def get_best_model(data, numeric_features, categorical_features, target_column):
    """
    Train multiple models and return the best performing one.
    
    Args:
        data (DataFrame): Dataset
        numeric_features (list): List of numeric feature names
        categorical_features (list): List of categorical feature names
        target_column (str): Target column name
        
    Returns:
        Pipeline: Best trained model pipeline
    """
    # Validate features exist in data
    all_features = numeric_features + categorical_features
    missing_features = [f for f in all_features if f not in data.columns]
    if missing_features:
        print(f"Warning: Features {missing_features} not found in dataset. Removing them.")
        numeric_features = [f for f in numeric_features if f in data.columns]
        categorical_features = [f for f in categorical_features if f in data.columns]
    
    # Make sure both numeric and categorical features are properly typed
    for feature in numeric_features:
        # Try to convert categorical features to numeric if possible
        try:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
            data[feature].fillna(data[feature].mean(), inplace=True)
        except:
            print(f"Warning: Could not convert {feature} to numeric. Using as-is.")
    
    # Convert categorical features to string to ensure proper encoding
    for feature in categorical_features:
        data[feature] = data[feature].astype(str)
    
    # Separate features and target
    X = data.drop([target_column], axis=1)
    y = data[target_column]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Check class balance
    class_counts = y.value_counts(normalize=True)
    class_balance = min(class_counts) / max(class_counts)
    use_f1 = class_balance > 0.5  # If class balance is high, use F1 Score, otherwise use Accuracy

    # Preprocessing pipelines with better error handling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # More robust to outliers than mean
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  # Better than 'most_frequent' for unknown values
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # sparse=False to avoid issues with some models
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified, more explicit
    )

    # Apply SMOTE for balancing if we have enough samples
    use_smote = len(y_train) > 10  # Only use SMOTE if we have enough samples
    
    # Define models with better default parameters
    models_with_weights = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, 
                                                solver='liblinear', C=1.0),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, 
                                              min_samples_split=5, min_samples_leaf=2),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                       max_depth=3),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    }

    results = {}
    best_models = {}

    for model_name, model in models_with_weights.items():
        try:
            # Use SMOTE only if we have enough samples
            if use_smote:
                smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
                pipeline = ImbPipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('smote', smote),
                    ('classifier', model)
                ])
            else:
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Store results
            results[model_name] = {
                'Accuracy': accuracy,
                'F1 Score': f1
            }
            best_models[model_name] = pipeline
            print(f"  ✅ Successfully trained {model_name}: Acc={accuracy:.2f}, F1={f1:.2f}")
        except Exception as e:
            print(f"  ⚠️ Error training {model_name}: {str(e)}")

    if not results:
        # If all models failed, return a simple fallback model
        print("  ⚠️ All models failed. Creating a simple fallback model.")
        fallback_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(solver='liblinear', max_iter=200))
        ])
        try:
            fallback_model.fit(X_train, y_train)
            return fallback_model
        except Exception as e:
            print(f"  ❌ Even fallback model failed: {str(e)}")
            # Create a dummy model that always predicts the most common class
            most_common_class = y_train.mode()[0]
            
            class DummyModel:
                def predict(self, X):
                    return np.array([most_common_class] * len(X))
                
                def predict_proba(self, X):
                    probs = np.zeros((len(X), len(np.unique(y))))
                    probs[:, 0] = 1.0  # Always predict class 0 with 100% confidence
                    return probs
            
            return DummyModel()

    # Select best model based on class balance condition
    best_metric = 'F1 Score' if use_f1 else 'Accuracy'
    best_model = max(results, key=lambda x: results[x][best_metric])
    print(f"  🏆 Best model: {best_model} with {best_metric} = {results[best_model][best_metric]:.2f}")

    return best_models[best_model]


class ModelWrapper:
    """
    Wrapper for ML models to handle unexpected inputs gracefully.
    """
    
    def __init__(self, model, feature_types):
        """
        Initialize the wrapper.
        
        Args:
            model: The sklearn pipeline or model to wrap
            feature_types: Dictionary mapping feature names to their types ('numeric' or 'categorical')
        """
        self.model = model
        self.feature_types = feature_types
        
    def predict(self, X):
        """
        Make predictions with better error handling.
        
        Args:
            X: Input features dataframe
            
        Returns:
            Predictions array
        """
        try:
            # Ensure proper feature types
            X_processed = X.copy()
            
            for feature, type_ in self.feature_types.items():
                if feature in X_processed.columns:
                    if type_ == 'numeric':
                        X_processed[feature] = pd.to_numeric(X_processed[feature], errors='coerce')
                        X_processed[feature].fillna(0, inplace=True)
                    else:
                        # Handle yes/no conversions for categorical features
                        if str(X_processed[feature].iloc[0]).lower() in ['yes', 'y', 'true', 't']:
                            X_processed[feature] = "1"
                        elif str(X_processed[feature].iloc[0]).lower() in ['no', 'n', 'false', 'f']:
                            X_processed[feature] = "0"
                        
                        # Convert to string for all categorical features
                        X_processed[feature] = X_processed[feature].astype(str)
            
            return self.model.predict(X_processed)
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return safe fallback prediction
            return np.array(['unknown'] * len(X)) 