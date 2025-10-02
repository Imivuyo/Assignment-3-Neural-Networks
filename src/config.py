# config.py
# Configuration settings for the experiments

CONFIG = {
    'hidden_units_to_test': [5, 10, 20, 30, 50],
    'n_runs': 30,  # For statistical validity; following ML comparison literature standards (e.g., Demšar, 2006)
    'max_epochs': 500,
    'early_stopping_patience': 50,  # Conservative choice to ensure convergence; typical range 10-50 (StackExchange, 2016)
    'train_val_test_split': (0.7, 0.15, 0.15),
    'random_seed': 42,
    'batch_size': 32,  # Added for mini-batch SGD; common for small-medium datasets
    
    # Hyperparameter search spaces; based on original papers and common practices
    'sgd_params': {  # Common ranges in optimization literature
        'learning_rate': [0.001, 0.01, 0.1],
        'momentum': [0.0, 0.5, 0.9]
    },
    'scg_params': {  # From Møller (1993) Scaled Conjugate Gradient
        'sigma': [1e-4, 1e-5, 1e-6],
        'lambda_': [1e-5, 1e-6, 1e-7]
    },
    'leapfrog_params': {
        'delta_t': [0.3, 0.5, 0.7],    # Small variation around standard 0.5
        'delta': [0.8, 1.0, 1.2],      # Small variation around standard 1.0
        # Keep all others fixed at standard values
        'm': [3],
        'delta_1': [0.001], 
        'j': [2],
        'M': [10],
        'N_max': [2]
    }
}