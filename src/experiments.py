import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import random
from scipy.stats import chi2, f, studentized_range, friedmanchisquare
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing, load_diabetes
import tensorflow as tf

from config import CONFIG
from data_utils import load_and_preprocess_data
from nn_model import FeedforwardNN
from trainers import SGDTrainer, SCGTrainer, LeapFrogTrainer

def find_optimal_hidden_units(dataset_name):
    """Find optimal number of hidden units by averaging validation loss across all algorithms"""
    print(f"\n{'='*70}")
    print(f"Finding optimal hidden units for {dataset_name} (averaging across algorithms)")
    print(f"{'='*70}")
    
    X_train, X_val, X_test, y_train, y_val, y_test, problem_type = load_and_preprocess_data(dataset_name)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
    
    algorithms = ['sgd', 'scg', 'leapfrog']
    results = []
    
    for hidden_units in CONFIG['hidden_units_to_test']:
        print(f"\nTesting {hidden_units} hidden units...")
        
        algo_val_losses = {algo: [] for algo in algorithms}
        
        for algo in algorithms:
            for run in range(8):
                np.random.seed(CONFIG['random_seed'] + run)
                tf.random.set_seed(CONFIG['random_seed'] + run)
                random.seed(CONFIG['random_seed'] + run)
                
                model = FeedforwardNN(input_dim, hidden_units, output_dim, problem_type)
                
                if algo == 'sgd':
                    trainer = SGDTrainer(learning_rate=0.01, momentum=0.9)
                elif algo == 'scg':
                    trainer = SCGTrainer()
                elif algo == 'leapfrog':
                    trainer = LeapFrogTrainer()
                
                history = trainer.train(model, X_train, y_train, X_val, y_val)
                assert 'train_loss_epochs' in history and 'val_loss_epochs' in history and 'epochs' in history, f"Invalid history keys for {algo}"
                
                algo_val_losses[algo].append(min(history['val_loss_epochs']) if history['val_loss_epochs'] else np.nan)
            
            mean_algo_loss = np.nanmean(algo_val_losses[algo])
            std_algo_loss = np.nanstd(algo_val_losses[algo])
            print(f"  {algo.upper()}: Mean val loss {mean_algo_loss:.4f} ± {std_algo_loss:.4f}")
        
        avg_mean_val_loss = np.nanmean([np.nanmean(algo_val_losses[algo]) for algo in algorithms])
        avg_std_val_loss = np.nanmean([np.nanstd(algo_val_losses[algo]) for algo in algorithms])
        
        results.append({
            'hidden_units': hidden_units,
            'mean_val_loss': avg_mean_val_loss,
            'std_val_loss': avg_std_val_loss
        })
        
    results_df = pd.DataFrame(results)
    optimal_idx = results_df['mean_val_loss'].idxmin()
    optimal_hidden_units = results_df.loc[optimal_idx, 'hidden_units']
    
    print(f"\n✓ Optimal hidden units: {optimal_hidden_units}")
    
    return optimal_hidden_units, results_df

def hyperparameter_search(dataset_name, algorithm, hidden_units):
    """Search for best hyperparameters with increased runs for reliability"""
    print(f"\n{'='*70}")
    print(f"Hyperparameter search: {dataset_name} - {algorithm.upper()}")
    print(f"{'='*70}")
    
    X_train, X_val, X_test, y_train, y_val, y_test, problem_type = load_and_preprocess_data(dataset_name)
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
    
    best_params = None
    best_val_loss = float('inf')
    results = []
    
    for run in range(10):
        np.random.seed(CONFIG['random_seed'] + run)
        tf.random.set_seed(CONFIG['random_seed'] + run)
        random.seed(CONFIG['random_seed'] + run)
    
        if algorithm == 'sgd':
            param_grid = CONFIG['sgd_params']
            for lr in param_grid['learning_rate']:
                for mom in param_grid['momentum']:
                    print(f"Testing lr={lr}, momentum={mom}")
                    
                    val_losses = []
                    for _ in range(10):
                        model = FeedforwardNN(input_dim, hidden_units, output_dim, problem_type)
                        trainer = SGDTrainer(learning_rate=lr, momentum=mom)
                        history = trainer.train(model, X_train, y_train, X_val, y_val)
                        assert 'val_loss_epochs' in history, "Missing val_loss_epochs in SGD history"
                        val_losses.append(min(history['val_loss_epochs']) if history['val_loss_epochs'] else np.nan)
                    
                    mean_val_loss = np.nanmean(val_losses)
                    std_val_loss = np.nanstd(val_losses)
                    results.append({'lr': lr, 'momentum': mom, 'mean_val_loss': mean_val_loss, 'std_val_loss': std_val_loss})
                    
                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        best_params = {'learning_rate': lr, 'momentum': mom}
        
        elif algorithm == 'scg':
            param_grid = CONFIG['scg_params']
            for sigma in param_grid['sigma']:
                for lambda_ in param_grid['lambda_']:
                    print(f"Testing sigma={sigma}, lambda_={lambda_}")
                    
                    val_losses = []
                    for _ in range(10):
                        model = FeedforwardNN(input_dim, hidden_units, output_dim, problem_type)
                        trainer = SCGTrainer(sigma=sigma, lambda_=lambda_)
                        history = trainer.train(model, X_train, y_train, X_val, y_val)
                        assert 'val_loss_epochs' in history, "Missing val_loss_epochs in SCG history"
                        val_losses.append(min(history['val_loss_epochs']) if history['val_loss_epochs'] else np.nan)
                    
                    mean_val_loss = np.nanmean(val_losses)
                    std_val_loss = np.nanstd(val_losses)
                    results.append({'sigma': sigma, 'lambda_': lambda_, 'mean_val_loss': mean_val_loss, 'std_val_loss': std_val_loss})
                    
                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        best_params = {'sigma': sigma, 'lambda_': lambda_}
        
        elif algorithm == 'leapfrog':
            param_grid = CONFIG['leapfrog_params']
            for delta_t in param_grid['delta_t']:
                for delta in param_grid['delta']:
                    print(f"Testing delta_t={delta_t}, delta={delta} (other parameters fixed: m=3, delta_1=0.001, j=2, M=10, N_max=2, epsilon=1e-5)")
                    
                    val_losses = []
                    for _ in range(10):
                        model = FeedforwardNN(input_dim, hidden_units, output_dim, problem_type)
                        trainer = LeapFrogTrainer(delta_t=delta_t, delta=delta, m=3, delta_1=0.001, j=2, M=10, N_max=2, epsilon=1e-5)
                        history = trainer.train(model, X_train, y_train, X_val, y_val)
                        assert 'val_loss_epochs' in history, "Missing val_loss_epochs in LeapFrog history"
                        val_losses.append(min(history['val_loss_epochs']) if history['val_loss_epochs'] else np.nan)
                    
                    mean_val_loss = np.nanmean(val_losses)
                    std_val_loss = np.nanstd(val_losses)
                    results.append({
                        'delta_t': delta_t,
                        'delta': delta,
                        'm': 3,
                        'delta_1': 0.001,
                        'j': 2,
                        'M': 10,
                        'N_max': 2,
                        'mean_val_loss': mean_val_loss,
                        'std_val_loss': std_val_loss
                    })
                    
                    if mean_val_loss < best_val_loss:
                        best_val_loss = mean_val_loss
                        best_params = {
                            'delta_t': delta_t,
                            'delta': delta,
                            'm': 3,
                            'delta_1': 0.001,
                            'j': 2,
                            'M': 10,
                            'N_max': 2
                        }
    
    print(f"\n✓ Best parameters: {best_params} (mean loss {best_val_loss:.4f})")
    
    return best_params

def run_final_comparison(dataset_name, configs):
    """Run final comparison with optimal settings, using in-memory aggregation for curves"""
    os.makedirs('results', exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Final Comparison: {dataset_name}")
    print(f"{'='*70}")
    
    X_train, X_val, X_test, y_train, y_val, y_test, problem_type = load_and_preprocess_data(dataset_name)
    
    preprocessing_details = {
        'california_housing': 'RobustScaler (features and target) due to extreme outliers (e.g., AveOccup max=1,243 vs. median=2.8) and censored target ($500,001 cap)',
        'diabetes': 'StandardScaler (features and target) for consistency, as data is pre-standardized',
        'fish_market': 'StandardScaler (numerical features), RobustScaler (target) due to right-skew (mean/median=1.46), one-hot encoding (species), removed one zero-weight error',
        'mnist': 'StandardScaler (features), one-hot encoding (targets), stratified split',
        'fashion_mnist': 'Pixel normalization ([0,255] to [0,1]), StandardScaler (features), one-hot encoding (targets), stratified split',
        'wine': 'StandardScaler (features) due to extreme scale differences (e.g., proline range=1,402), one-hot encoding (targets), stratified split'
    }
    print(f"Preprocessing: {preprocessing_details.get(dataset_name, 'Unknown')}")
    
    target_skew = None
    outlier_ratio = None
    if problem_type == 'regression':
        if dataset_name == 'california_housing':
            data = fetch_california_housing()
            raw_y = data.target
        elif dataset_name == 'diabetes':
            data = load_diabetes()
            raw_y = data.target
        elif dataset_name == 'fish_market':
            df = pd.read_csv('../../data/Fish.csv')
            df = df[df['Weight'] > 0]
            raw_y = df['Weight'].values
        else:
            raw_y = y_train.flatten()
        
        target_skew = np.mean(raw_y) / np.median(raw_y) if np.median(raw_y) != 0 else np.nan
        print(f"Target skew (mean/median): {target_skew:.2f}")
        
        iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)
        median = np.median(X_train, axis=0)
        outliers = np.any((X_train < (median - 3 * iqr)) | (X_train > (median + 3 * iqr)), axis=1)
        outlier_ratio = np.mean(outliers) * 100
        print(f"Feature outlier ratio (>3×IQR): {outlier_ratio:.2f}%")
    
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
    
    all_results = []
    aggregated_curves = {}
    
    for algo_name, config in configs.items():
        print(f"\nRunning {algo_name.upper()}...")
        hidden_units = config['hidden_units']
        params = config['params']
        
        train_losses_runs = []
        val_losses_runs = []
        
        for run in range(CONFIG['n_runs']):
            if run % 5 == 0:
                print(f"  Run {run+1}/{CONFIG['n_runs']}")
            
            np.random.seed(CONFIG['random_seed'] + run)
            tf.random.set_seed(CONFIG['random_seed'] + run)
            random.seed(CONFIG['random_seed'] + run)
            
            start_time = time.time()
            model = FeedforwardNN(input_dim, hidden_units, output_dim, problem_type)
            
            if algo_name == 'sgd':
                trainer = SGDTrainer(**params)
            elif algo_name == 'scg':
                trainer = SCGTrainer(**params)
            elif algo_name == 'leapfrog':
                trainer = LeapFrogTrainer(**{**params, 'epsilon': 1e-5})
            
            history = trainer.train(model, X_train, y_train, X_val, y_val)
            assert 'train_loss_epochs' in history and 'val_loss_epochs' in history and 'epochs' in history, f"Invalid history keys for {algo_name}"
            
            training_time = time.time() - start_time
            
            y_pred = model.predict(X_test)
            if problem_type == 'classification':
                y_test_classes = np.argmax(y_test, axis=1)
                test_accuracy = accuracy_score(y_test_classes, y_pred)
                test_f1 = f1_score(y_test_classes, y_pred, average='macro')
                primary_metric = test_accuracy
                primary_metric_name = 'accuracy'
                supporting_metrics = {'f1_macro': test_f1}
            else:
                test_mse = mean_squared_error(y_test.flatten(), y_pred)
                test_rmse = np.sqrt(test_mse)
                test_mae = mean_absolute_error(y_test.flatten(), y_pred)
                test_r2 = r2_score(y_test.flatten(), y_pred)
                primary_metric = test_rmse
                primary_metric_name = 'rmse'
                supporting_metrics = {'mae': test_mae, 'r2': test_r2}
            
            if history['val_loss_epochs']:
                min_val_loss = min(history['val_loss_epochs'])
                try:
                    convergence_epoch = min(i for i, loss in enumerate(history['val_loss_epochs']) if loss <= 1.01 * min_val_loss) + 1
                except ValueError:
                    convergence_epoch = history['epochs']
            else:
                convergence_epoch = history['epochs']
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"  Warning: {algo_name} run {run+1} converged early (gradient norm < 1e-5 or early stopping) with params: {param_str}, epsilon=1e-5")
            
            run_data = {
                'algorithm': algo_name,
                'run': run,
                'primary_metric': primary_metric,
                'metric_name': primary_metric_name,
                'training_time': training_time,
                'epochs': history['epochs'],
                'convergence_epoch': convergence_epoch,
                'final_train_loss': history['train_loss_epochs'][-1] if history['train_loss_epochs'] else np.nan,
                'final_val_loss': history['val_loss_epochs'][-1] if history['val_loss_epochs'] else np.nan,
                **supporting_metrics
            }
            if problem_type == 'regression':
                run_data.update({'target_skew': target_skew, 'outlier_ratio': outlier_ratio})
            all_results.append(run_data)
            
            train_losses_runs.append(np.array(history['train_loss_epochs']) if history['train_loss_epochs'] else np.array([]))
            val_losses_runs.append(np.array(history['val_loss_epochs']) if history['val_loss_epochs'] else np.array([]))
        
        if train_losses_runs and val_losses_runs:
            lengths = [len(arr) for arr in train_losses_runs if len(arr) > 0]
            max_epochs_algo = max(lengths) if lengths else 0
            if max_epochs_algo > 0:
                padded_train = np.array([
                    np.pad(arr, (0, max_epochs_algo - len(arr)), 'constant', constant_values=np.nan) if len(arr) > 0 else np.full(max_epochs_algo, np.nan)
                    for arr in train_losses_runs
                ])
                padded_val = np.array([
                    np.pad(arr, (0, max_epochs_algo - len(arr)), 'constant', constant_values=np.nan) if len(arr) > 0 else np.full(max_epochs_algo, np.nan)
                    for arr in val_losses_runs
                ])
                
                mean_train = np.nanmean(padded_train, axis=0)
                std_train = np.nanstd(padded_train, axis=0)
                mean_val = np.nanmean(padded_val, axis=0)
                std_val = np.nanstd(padded_val, axis=0)
                
                aggregated_curves[algo_name] = {
                    'epochs': np.arange(max_epochs_algo),
                    'mean_train_loss': mean_train,
                    'std_train_loss': std_train,
                    'mean_val_loss': mean_val,
                    'std_val_loss': std_val
                }
            else:
                print(f"  Warning: No valid curves for {algo_name} (all runs converged early)")
                aggregated_curves[algo_name] = {
                    'epochs': np.array([]),
                    'mean_train_loss': np.array([]),
                    'std_train_loss': np.array([]),
                    'mean_val_loss': np.array([]),
                    'std_val_loss': np.array([])
                }
        else:
            print(f"  Warning: No curves collected for {algo_name} (all runs empty)")
            aggregated_curves[algo_name] = {
                'epochs': np.array([]),
                'mean_train_loss': np.array([]),
                'std_train_loss': np.array([]),
                'mean_val_loss': np.array([]),
                'std_val_loss': np.array([])
            }
        
        agg_df = pd.DataFrame(aggregated_curves[algo_name])
        run_dir = f'results/{dataset_name}_epoch_losses'
        os.makedirs(run_dir, exist_ok=True)
        agg_df.to_csv(f'{run_dir}/{algo_name}_aggregated.csv', index=False)
        print(f"  ✓ Aggregated curves saved for {algo_name}")
    
    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print(f"Warning: No results collected for {dataset_name}")
        return results_df, problem_type
    
    results_df.to_csv(f'results/{dataset_name}_final_comparison.csv', index=False)
    
    print(f"\n{'='*70}")
    print(f"Summary for {dataset_name}")
    print(f"{'='*70}")
    for algo in configs.keys():
        algo_results = results_df[results_df['algorithm'] == algo]
        if algo_results.empty:
            print(f"  Warning: No results for {algo}")
            continue
        mean_metric = algo_results['primary_metric'].mean()
        std_metric = algo_results['primary_metric'].std()
        ci_95 = 1.96 * std_metric / np.sqrt(len(algo_results)) if len(algo_results) > 0 else np.nan
        mean_time = algo_results['training_time'].mean()
        mean_conv = algo_results['convergence_epoch'].mean()
        if algo_results['metric_name'].iloc[0] == 'accuracy':
            mean_f1 = algo_results['f1_macro'].mean()
            print(f"{algo.upper():12s}: accuracy={mean_metric:.4f}±{ci_95:.4f}, f1_macro={mean_f1:.4f}, time={mean_time:.2f}s, conv_epochs={mean_conv:.1f}")
        else:
            mean_mae = algo_results['mae'].mean()
            mean_r2 = algo_results['r2'].mean()
            print(f"{algo.upper():12s}: rmse={mean_metric:.4f}±{ci_95:.4f}, mae={mean_mae:.4f}, r2={mean_r2:.4f}, time={mean_time:.2f}s, conv_epochs={mean_conv:.1f}")
    
    generate_visualizations(dataset_name, results_df, configs, aggregated_curves)
    
    return results_df, problem_type

def generate_visualizations(dataset_name, results_df, configs, aggregated_curves):
    """Generate visualizations: boxplots, learning curves using aggregated curves"""
    os.makedirs(f'results/{dataset_name}_plots', exist_ok=True)
    
    if results_df.empty:
        print(f"Warning: Skipping visualizations for {dataset_name} due to empty results")
        return
    
    fig, ax = plt.subplots()
    data = [results_df[results_df['algorithm'] == algo]['primary_metric'] for algo in configs.keys()]
    valid_data = [d for d in data if len(d) > 0]
    valid_labels = [algo for algo, d in zip(configs.keys(), data) if len(d) > 0]
    if valid_data:
        ax.boxplot(valid_data, labels=valid_labels)
        ax.set_title(f'Test Performance Boxplot - {dataset_name}')
        ax.set_ylabel(results_df['metric_name'].iloc[0].capitalize())
        plt.savefig(f'results/{dataset_name}_plots/test_metric_boxplot.png')
        plt.close()
    else:
        print(f"Warning: No valid data for boxplot in {dataset_name}")
    
    fig, ax = plt.subplots()
    for algo in configs.keys():
        agg_data = aggregated_curves.get(algo, {})
        if len(agg_data.get('epochs', [])) > 0:
            epochs = agg_data['epochs']
            mean_val = agg_data['mean_val_loss']
            std_val = agg_data['std_val_loss']
            
            ax.plot(epochs, mean_val, label=algo.upper())
            ax.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2)
    
    if aggregated_curves:
        ax.set_title(f'Average Validation Learning Curves - {dataset_name}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Validation Loss (Huber)' if results_df['metric_name'].iloc[0] == 'rmse' else 'Validation Loss (Cross-Entropy)')
        ax.legend()
        plt.savefig(f'results/{dataset_name}_plots/learning_curves.png')
        plt.close()

def create_summary_table(all_results):
    """Create summary table aggregating performance across all datasets"""
    algorithms = ['sgd', 'scg', 'leapfrog']
    datasets = list(all_results.keys())
    summary_data = []
    
    classification_datasets = [d for d, v in all_results.items() if v['problem_type'] == 'classification']
    regression_datasets = [d for d, v in all_results.items() if v['problem_type'] == 'regression']
    
    for algo in algorithms:
        primary_metrics = []
        training_times = []
        convergence_epochs = []
        f1_macros = []
        maes = []
        r2s = []
        
        for dataset in datasets:
            results_df = all_results[dataset]['results_df']
            algo_results = results_df[results_df['algorithm'] == algo]
            if algo_results.empty:
                continue
            primary_metrics.extend(algo_results['primary_metric'].values)
            training_times.extend(algo_results['training_time'].values)
            convergence_epochs.extend(algo_results['convergence_epoch'].values)
            if all_results[dataset]['problem_type'] == 'classification':
                f1_macros.extend(algo_results['f1_macro'].values)
            else:
                maes.extend(algo_results['mae'].values)
                r2s.extend(algo_results['r2'].values)
        
        mean_metric = np.nanmean(primary_metrics) if primary_metrics else np.nan
        std_metric = np.nanstd(primary_metrics) if primary_metrics else np.nan
        ci_95_metric = 1.96 * std_metric / np.sqrt(len(primary_metrics)) if primary_metrics else np.nan
        
        mean_time = np.nanmean(training_times) if training_times else np.nan
        std_time = np.nanstd(training_times) if training_times else np.nan
        ci_95_time = 1.96 * std_time / np.sqrt(len(training_times)) if training_times else np.nan
        
        mean_conv = np.nanmean(convergence_epochs) if convergence_epochs else np.nan
        std_conv = np.nanstd(convergence_epochs) if convergence_epochs else np.nan
        ci_95_conv = 1.96 * std_conv / np.sqrt(len(convergence_epochs)) if convergence_epochs else np.nan
        
        summary_row = {
            'algorithm': algo,
            'mean_primary_metric': mean_metric,
            'ci_95_primary_metric': ci_95_metric,
            'mean_training_time': mean_time,
            'ci_95_training_time': ci_95_time,
            'mean_convergence_epoch': mean_conv,
            'ci_95_convergence_epoch': ci_95_conv
        }
        if f1_macros and classification_datasets:
            summary_row['mean_f1_macro'] = np.nanmean(f1_macros)
        if maes and regression_datasets:
            summary_row['mean_mae'] = np.nanmean(maes)
            summary_row['mean_r2'] = np.nanmean(r2s)
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def perform_statistical_tests(all_results_dict):
    """Perform Friedman test and Nemenyi post-hoc across all datasets"""
    algorithms = list(next(iter(all_results_dict.values()))['configs'].keys())
    datasets = list(all_results_dict.keys())
    N = len(datasets)
    k = len(algorithms)
    
    metrics = np.zeros((N, k))
    for d_idx, dataset in enumerate(datasets):
        results_df = all_results_dict[dataset]['results_df']
        problem_type = all_results_dict[dataset]['problem_type']
        for a_idx, algo in enumerate(algorithms):
            algo_results = results_df[results_df['algorithm'] == algo]
            if algo_results.empty:
                metrics[d_idx, a_idx] = np.nan
            else:
                mean_metric = algo_results['primary_metric'].mean()
                metrics[d_idx, a_idx] = mean_metric if problem_type == 'classification' else -mean_metric
    
    print("Metrics matrix:\n", metrics)
    
    ranks = np.zeros((N, k))
    for d in range(N):
        valid_metrics = metrics[d]
        if np.all(np.isnan(valid_metrics)):
            ranks[d] = np.full(k, np.nan)
        else:
            ranks[d] = np.argsort(np.argsort(-np.nan_to_num(valid_metrics, nan=np.nanmax(valid_metrics) + 1))) + 1
    
    print("Ranks matrix:\n", ranks)
    
    alg_metrics = [metrics[:, j] for j in range(k)]
    valid_algs = [j for j in range(k) if not np.all(np.isnan(alg_metrics[j]))]
    if len(valid_algs) < 2:
        print("Warning: Insufficient valid algorithms for Friedman test")
        summary = {
            'avg_ranks': dict(zip(algorithms, [np.nan] * k)),
            'friedman_p': np.nan,
            'nemenyi_cd': np.nan,
            'wdl': {algo: {'wins': 0, 'draws': 0, 'losses': 0} for algo in algorithms}
        }
        with open('results/overall_statistical_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        return summary
    
    valid_metrics = [alg_metrics[j] for j in valid_algs]
    valid_algorithms = [algorithms[j] for j in valid_algs]
    stat, p_value = friedmanchisquare(*valid_metrics)
    
    R = np.nansum(ranks, axis=0)
    avg_ranks = R / np.sum(~np.isnan(ranks), axis=0)
    
    q_alpha = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    
    wdl = {algo: {'wins': 0, 'draws': 0, 'losses': 0} for algo in algorithms}
    for i in range(k):
        for j in range(i+1, k):
            if np.isnan(avg_ranks[i]) or np.isnan(avg_ranks[j]):
                continue
            rank_diff = abs(avg_ranks[i] - avg_ranks[j])
            if rank_diff > cd:
                better = i if avg_ranks[i] < avg_ranks[j] else j
                worse = j if better == i else i
                wdl[algorithms[better]]['wins'] += 1
                wdl[algorithms[worse]]['losses'] += 1
            else:
                wdl[algorithms[i]]['draws'] += 1
                wdl[algorithms[j]]['draws'] += 1
    
    summary = {
        'avg_ranks': dict(zip(algorithms, avg_ranks.tolist())),
        'friedman_p': float(p_value),
        'nemenyi_cd': float(cd),
        'wdl': wdl
    }
    with open('results/overall_statistical_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(1, k)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Average Rank')
    sorted_idx = np.argsort(np.nan_to_num(avg_ranks, nan=np.inf))
    sorted_algos = np.array(algorithms)[sorted_idx]
    sorted_ranks = avg_ranks[sorted_idx]
    valid_idx = [i for i, r in enumerate(sorted_ranks) if not np.isnan(r)]
    if valid_idx:
        valid_ranks = sorted_ranks[valid_idx]
        valid_algos = sorted_algos[valid_idx]
        ax.plot(valid_ranks, [0.5]*len(valid_ranks), marker='o')
        for i, (rank, algo) in enumerate(zip(valid_ranks, valid_algos)):
            ax.text(rank, 0.6, algo.upper(), rotation=45)
        
        for i in range(len(valid_ranks)-1):
            if abs(valid_ranks[i+1] - valid_ranks[i]) <= cd:
                ax.hlines(0.4, valid_ranks[i], valid_ranks[i+1], lw=2)
    
    ax.set_title('Critical Difference Diagram (Nemenyi test)')
    plt.savefig('results/cd_diagram.png')
    plt.close()
    
    print("\nOverall Statistical Summary:")
    print(f"Friedman p-value: {p_value:.4f}")
    print(f"Nemenyi CD (alpha=0.05): {cd:.4f}")
    print("Algorithm rankings (lower better):", dict(zip(algorithms, avg_ranks.tolist())))
    print("Win-Draw-Loss:", wdl)
    
    return summary