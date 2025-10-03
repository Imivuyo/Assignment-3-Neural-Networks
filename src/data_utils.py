# data_utils.py
# Data loading and preprocessing functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_wine, load_diabetes, fetch_california_housing
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

from config import CONFIG

def load_and_preprocess_data(dataset_name):
    """
    Load and preprocess datasets with appropriate transformations.
    Handles outliers, missing values, consistent scaling, and categorical encoding.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, problem_type
    """
    np.random.seed(CONFIG['random_seed'])
    
    # Load dataset

    if dataset_name == 'mnist':
        data = load_digits()
        X, y = data.data, data.target
        problem_type = 'classification'
        n_classes = 10

        # Standardize
        X = StandardScaler().fit_transform(X)
        y = np.eye(n_classes)[y]

    elif dataset_name == 'fashion_mnist':
        (X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

        X_train_full = X_train_full.reshape(-1, 28*28).astype('float32') / 255.0
        X_test_full = X_test_full.reshape(-1, 28*28).astype('float32') / 255.0

        # Standardize
        scaler = StandardScaler()
        X_train_full = scaler.fit_transform(X_train_full)
        X_test_full = scaler.transform(X_test_full)
        problem_type = 'classification'
        n_classes = 10
        y_train_full = np.eye(n_classes)[y_train_full]
        y_test_full = np.eye(n_classes)[y_test_full]

        # Combine for consistent splitting
        X = np.vstack([X_train_full, X_test_full])
        y = np.vstack([y_train_full, y_test_full])

    elif dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        problem_type = 'classification'
        n_classes = 3

        # Standardize
        X = StandardScaler().fit_transform(X)
        y = np.eye(n_classes)[y]

    elif dataset_name == 'diabetes':
        data = load_diabetes()

        X = StandardScaler().fit_transform(data.data)
        y = StandardScaler().fit_transform(data.target.reshape(-1, 1))
        problem_type = 'regression'

    elif dataset_name == 'california_housing':
        data = fetch_california_housing()
        X = RobustScaler().fit_transform(data.data)
        y = RobustScaler().fit_transform(data.target.reshape(-1, 1))
        problem_type = 'regression'

    elif dataset_name == 'fish_market':
        try:
            df = pd.read_csv('../../data/Fish.csv')

            # Remove zero-weight fish (data error)
            if df['Weight'].min() == 0:
                df = df[df['Weight'] > 0]

            # Encode 'Species' with OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            species_encoded = encoder.fit_transform(df[['Species']])

            # Use numerical features + encoded species
            X_numerical = df[['Length1', 'Length2', 'Length3', 'Height', 'Width']].values
            X_numerical = StandardScaler().fit_transform(X_numerical)

            X = np.hstack([X_numerical, species_encoded])

            # Use RobustScaler for target due to right-skewed distribution
            y = RobustScaler().fit_transform(df['Weight'].values.reshape(-1, 1))

            problem_type = 'regression'
        except FileNotFoundError:
            raise FileNotFoundError("Fish Market dataset '../data/Fish.csv' not found. Please ensure the file exists.")

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if problem_type == 'classification':
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=CONFIG['random_seed'], stratify=np.argmax(y, axis=1) if y.ndim > 1 else y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=CONFIG['random_seed'], stratify=np.argmax(y_temp, axis=1) if y_temp.ndim > 1 else y_temp
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=CONFIG['random_seed']
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=CONFIG['random_seed']
        )

    print(f"\nDataset: {dataset_name}")
    print(f"Problem type: {problem_type}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"X min/max: {X.min():.4f}/{X.max():.4f}")
    if problem_type == 'regression':
        print(f"y min/max: {y.min():.4f}/{y.max():.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test, problem_type