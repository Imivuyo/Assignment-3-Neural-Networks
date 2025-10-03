import os
import json
import pandas as pd
from multiprocessing import Pool
from experiments import (
    find_optimal_hidden_units,
    hyperparameter_search,
    run_final_comparison,
    perform_statistical_tests,
    create_summary_table
)

# -------------------------------------------------------------------------
# CONFIG: Choose datasets for testing vs full run
# -------------------------------------------------------------------------
ALL_DATASETS = ['mnist', 'fashion_mnist', 'wine', 'diabetes', 'california_housing', 'fish_market']

# Toggle here:
DEBUG_MODE = False   # set to False when running the full pipeline

if DEBUG_MODE:
    datasets = ['wine']  # small test run
    print("DEBUG MODE ON: Running only on:", datasets)
else:
    datasets = ALL_DATASETS
    print("FULL RUN MODE: Running on all datasets")

# -------------------------------------------------------------------------
# Step 1: Architecture selection
# -------------------------------------------------------------------------
optimal_architectures = {}
for dataset in datasets:
    print(f"\nProcessing {dataset}...")
    optimal_hidden_units, _ = find_optimal_hidden_units(dataset)
    optimal_architectures[dataset] = optimal_hidden_units

# Save optimal architectures
arch_df = pd.DataFrame([optimal_architectures], index=[0])
os.makedirs('results', exist_ok=True)
arch_df.to_csv('results/optimal_architectures.csv', index=False)
print("\n✓ Optimal architectures saved to results/optimal_architectures.csv")

# -------------------------------------------------------------------------
# Step 2: Hyperparameter tuning (parallelized)
# -------------------------------------------------------------------------
def tune_hyperparams(args):
    dataset, algo, hidden_units = args
    try:
        print(f"\nTuning hyperparameters for {algo} on {dataset}...")
        best_params = hyperparameter_search(dataset, algo, hidden_units)
        return dataset, algo, best_params
    except Exception as e:
        print(f"ERROR in tune_hyperparams for {dataset}-{algo}: {e}")
        import traceback
        traceback.print_exc()
        return dataset, algo, {}

best_hyperparams = {dataset: {} for dataset in datasets}
with Pool(processes=10) as pool:  # Use your 4 cores
    results = pool.map(
        tune_hyperparams,
        [(dataset, algo, optimal_architectures[dataset])
         for dataset in datasets
         for algo in ['sgd', 'scg', 'leapfrog']]
    )

for dataset, algo, best_params in results:
    best_hyperparams[dataset][algo] = best_params

# Save best hyperparameters
hyperparam_rows = []
for dataset in datasets:
    for algo in best_hyperparams[dataset]:
        row = {'dataset': dataset, 'algorithm': algo}
        row.update(best_hyperparams[dataset][algo])
        hyperparam_rows.append(row)
hyperparam_df = pd.DataFrame(hyperparam_rows)
hyperparam_df.to_csv('results/best_hyperparameters.csv', index=False)
print("\n✓ Best hyperparameters saved to results/best_hyperparameters.csv")

# -------------------------------------------------------------------------
# Step 3: Final comparison (parallelized)
# -------------------------------------------------------------------------
def run_comparison(args):
    dataset, configs = args
    try:
        results_df, problem_type = run_final_comparison(dataset, configs)
        return dataset, results_df, problem_type, configs
    except Exception as e:
        print(f"ERROR in run_comparison for {dataset}: {e}")
        import traceback
        traceback.print_exc()
        empty_df = pd.DataFrame()
        return dataset, empty_df, 'unknown', configs

with Pool(processes=10) as pool:
    results = pool.map(
        run_comparison,
        [(dataset, {
            'sgd': {'hidden_units': optimal_architectures[dataset],
                    'params': best_hyperparams[dataset]['sgd']},
            'scg': {'hidden_units': optimal_architectures[dataset],
                    'params': best_hyperparams[dataset]['scg']},
            'leapfrog': {'hidden_units': optimal_architectures[dataset],
                         'params': best_hyperparams[dataset]['leapfrog']}
        }) for dataset in datasets]
    )

all_results = {}
for dataset, results_df, problem_type, configs in results:
    all_results[dataset] = {
        'results_df': results_df,
        'configs': configs,
        'problem_type': problem_type  # Use the returned problem_type
    }

# -------------------------------------------------------------------------
# Step 4: Statistical analysis
# -------------------------------------------------------------------------
perform_statistical_tests(all_results)

# -------------------------------------------------------------------------
# Step 5: Summary table
# -------------------------------------------------------------------------
summary_df = create_summary_table(all_results)
summary_df.to_csv('results/overall_summary.csv', index=False)
print("\n✓ Overall summary table saved to results/overall_summary.csv")