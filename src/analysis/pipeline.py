import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import joblib
import time

warnings.filterwarnings('ignore')

# Import feature selectors
from .feature_selectors import (
    BiologicalFeatureSelector,
    LassoFeatureSelector,
    StrainAwareSplitter
)

def run_full_analysis(data_path, timepoints=None, methods=None, 
                     output_dir='results', n_splits=200, n_jobs=-1,
                     verbose=True, quick_mode=False):
    if timepoints is None:
        timepoints = ['0min', '20min', '30min', '60min']
    if methods is None:
        methods = ['LDA', 'PLS_DA', 'SVM_linear', 'SVM_rbf']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    # Load data
    if verbose:
        print(f"Loading data from {data_path}...")
    
    try:
        df = pd.read_excel(data_path, index_col=0, engine='odf')
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you have odfpy installed: pip install odfpy")
        return None
    
    freq = list(df.columns[0:490].values)
    
    # Process targets
    df['simple_target'] = df['target'].apply(lambda x: 'MRSA' if 'MRSA' in x else 'MSSA')
    df['time_info'] = df['target'].apply(lambda x: re.sub(r'(MRSA|MSSA)\s*', '', x).strip())
    
    all_results = {}
    
    # Overall progress bar
    total_tasks = len(timepoints) * len(methods)
    overall_pbar = tqdm(total=total_tasks, desc="Overall Progress", 
                       disable=not verbose, position=0)
    
    for timepoint in timepoints:
        if verbose:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {timepoint}")
            print(f"{'='*60}")
        
        # Filter for timepoint
        time_groups = {
            '0min': ['0 min'],
            '20min': ['20 min'], 
            '30min': ['30 min'],
            '60min': ['60 min']
        }
        
        patterns = time_groups[timepoint]
        time_mask = np.zeros(len(df), dtype=bool)
        for pattern in patterns:
            if pattern == '0 min':
                exact_pattern_mask = df['time_info'].str.contains(r'\b0\s*min\b', case=False, na=False)
            else:
                exact_pattern_mask = df['time_info'].str.contains(pattern, case=False, na=False)
            time_mask = time_mask | exact_pattern_mask
        
        df_time = df[time_mask].copy()
        
        if len(df_time) == 0:
            if verbose:
                print(f"  No data available for {timepoint}")
            overall_pbar.update(len(methods))
            continue
        
        # Filter out control samples
        non_control_mask = ~df_time['target'].str.contains('control', case=False, na=False)
        df_time_eval = df_time[non_control_mask]
        
        total_samples = len(df_time_eval)
        
        if verbose:
            print(f"  Total non-control samples: {total_samples}")
        
        if total_samples < 13:
            if verbose:
                print(f"  Not enough non-control samples for {timepoint}")
            overall_pbar.update(len(methods))
            continue
        
        # Prepare data
        X = df_time_eval[freq].values
        y = df_time_eval['simple_target'].values
        target_labels = df_time_eval['target'].values
        
        if verbose:
            print(f"  Samples: {len(X)}")
            print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Generate splits
        splitter = StrainAwareSplitter(verbose=verbose)
        splits = splitter.generate_flexible_splits(
            X, y, target_labels, 
            min_train_samples=12, 
            max_train_samples=15
        )
        
        if not splits:
            if verbose:
                print("  No valid splits generated!")
            overall_pbar.update(len(methods))
            continue
        
        # Limit splits for quick mode
        if quick_mode:
            splits = splits[:min(5, len(splits))]
            n_splits = min(5, n_splits)
        
        # Process each method
        timepoint_results = {}
        
        for method in methods:
            if verbose:
                print(f"\n  Running {method}...")
            
            # Train and evaluate
            results = evaluate_method(
                X=X, y=y, freq=freq, splits=splits[:n_splits],
                method=method, timepoint=timepoint, 
                n_jobs=n_jobs, verbose=verbose
            )
            
            if results:
                timepoint_results[method] = results
                
                # Save model if successful
                if results['n_splits'] > 0:
                    save_trained_model(
                        X=X, y=y, freq=freq, timepoint=timepoint,
                        method=method, output_dir=output_dir
                    )
            
            overall_pbar.update(1)
        
        if timepoint_results:
            all_results[timepoint] = timepoint_results
            save_timepoint_results(timepoint_results, timepoint, output_dir)
    
    overall_pbar.close()
    
    # Create summary plot
    if all_results:
        create_summary_plot(all_results, output_dir, verbose)
    
    return all_results

def evaluate_method(X, y, freq, splits, method, timepoint, n_jobs=-1, verbose=True):
    biological_selector = BiologicalFeatureSelector()
    lasso_selector = LassoFeatureSelector()
    
    # Prepare arguments for parallel processing
    args_list = [(train_idx, test_idx, X, y, freq, method, timepoint,
                  biological_selector, lasso_selector) 
                for train_idx, test_idx in splits]
    
    if n_jobs == -1:
        n_jobs = min(mp.cpu_count(), len(splits))
    
    all_results = []
    
    # Process splits in parallel with progress bar
    with tqdm(total=len(args_list), desc=f"    {method} splits", 
             disable=not verbose, position=1, leave=False) as pbar:
        
        if n_jobs > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {executor.submit(process_split_wrapper, arg): i 
                          for i, arg in enumerate(args_list)}
                
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout
                        all_results.append(result)
                    except Exception as e:
                        if verbose:
                            print(f"    Split failed: {e}")
                        all_results.append({'success': False})
                    pbar.update(1)
        else:
            # Sequential processing (for debugging)
            for arg in args_list:
                try:
                    result = process_split_wrapper(arg)
                    all_results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"    Split failed: {e}")
                    all_results.append({'success': False})
                pbar.update(1)
    
    # Aggregate results
    accuracies, train_sizes, preselected, final_features = [], [], [], []
    y_true_all, y_pred_all = [], []
    successful = 0
    
    for result in all_results:
        if result['success']:
            accuracies.append(result['accuracy'])
            train_sizes.append(result['train_size'])
            preselected.append(result['preselected_features'])
            final_features.append(result['final_features'])
            y_true_all.extend(result['true_labels'])
            y_pred_all.extend(result['predictions'])
            successful += 1
    
    if not accuracies:
        if verbose:
            print(f"    No successful splits for {method}")
        return None
    
    # Calculate metrics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    n_splits = len(accuracies)
    
    if n_splits > 1:
        ci_lower = max(0.0, mean_acc - 1.96 * (std_acc / np.sqrt(n_splits)))
        ci_upper = min(1.0, mean_acc + 1.96 * (std_acc / np.sqrt(n_splits)))
    else:
        ci_lower = ci_upper = mean_acc
    
    size_counts = Counter(train_sizes)
    
    if verbose:
        print(f"    Successful: {successful}/{len(all_results)}")
        print(f"    Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"    95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"    Features: {np.mean(preselected):.0f} -> {np.mean(final_features):.0f}")
        print(f"    Train sizes: {dict(size_counts)}")
    
    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'confidence_interval': (ci_lower, ci_upper),
        'n_splits': n_splits,
        'all_accuracies': accuracies,
        'train_sizes': train_sizes,
        'preselected_features': preselected,
        'final_features': final_features,
        'train_size_distribution': dict(size_counts),
        'y_true': y_true_all,
        'y_pred': y_pred_all
    }

def process_split_wrapper(args):
    (train_idx, test_idx, X, y, freq, method, timepoint,
     biological_selector, lasso_selector) = args
    
    # Suppress prints in parallel processes
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Biological pre-selection
        X_train_bio, bio_indices = biological_selector.pre_select_features(
            X_train, freq, timepoint, verbose=False
        )
        X_test_bio = X_test[:, bio_indices]
        
        # LASSO selection
        selected_mask = lasso_selector.optimize_wavelengths(
            X_train_bio, y_train, n_bootstrap=50, 
            stability_threshold=0.1, verbose=False
        )
        
        if selected_mask is None or np.sum(selected_mask) == 0:
            X_train_selected = X_train_bio
            X_test_selected = X_test_bio
            n_features = X_train_bio.shape[1]
        else:
            X_train_selected = X_train_bio[:, selected_mask]
            X_test_selected = X_test_bio[:, selected_mask]
            n_features = np.sum(selected_mask)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train and predict
        if method == 'LDA':
            model = LinearDiscriminantAnalysis()
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        elif method == 'PLS_DA':
            y_train_numeric = np.where(y_train == 'MRSA', 1, 0)
            model = PLSRegression(n_components=min(3, X_train_scaled.shape[1]))
            model.fit(X_train_scaled, y_train_numeric)
            y_pred_proba = model.predict(X_test_scaled).flatten()
            pred = np.where(y_pred_proba > 0.5, 'MRSA', 'MSSA')
        elif method == 'SVM_linear':
            model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        elif method == 'SVM_rbf':
            model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        accuracy = accuracy_score(y_test, pred)
        
        result = {
            'test_indices': test_idx,
            'predictions': list(pred),
            'true_labels': list(y_test),
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'preselected_features': X_train_bio.shape[1],
            'final_features': n_features,
            'success': True
        }
        
    except Exception:
        result = {
            'test_indices': test_idx,
            'predictions': [],
            'true_labels': list(y_test),
            'accuracy': 0.0,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'success': False
        }
    finally:
        sys.stdout = old_stdout
    
    return result

# ... [rest of the functions remain the same, just remove any emojis] ...

def save_trained_model(X, y, freq, timepoint, method, output_dir):
    try:
        model_path = os.path.join(output_dir, 'models', f'{method}_{timepoint}.joblib')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Use your existing feature selection logic
        biological_selector = BiologicalFeatureSelector()
        lasso_selector = LassoFeatureSelector()
        
        # Apply feature selection
        X_bio, bio_indices = biological_selector.pre_select_features(
            X, freq, timepoint, verbose=False
        )
        
        selected_mask = lasso_selector.optimize_wavelengths(
            X_bio, y, n_bootstrap=50, 
            stability_threshold=0.1, verbose=False
        )
        
        if selected_mask is not None and np.sum(selected_mask) > 0:
            X_selected = X_bio[:, selected_mask]
            selected_freq = [freq[i] for i in bio_indices[selected_mask]]
        else:
            X_selected = X_bio
            selected_freq = [freq[i] for i in bio_indices]
        
        # Train model
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        if method == 'LDA':
            model = LinearDiscriminantAnalysis()
        elif method == 'PLS_DA':
            model = PLSRegression(n_components=min(3, X_scaled.shape[1]))
            y_numeric = np.where(y == 'MRSA', 1, 0)  # Use a different variable
            model.fit(X_scaled, y_numeric)  # Train with numeric y
        elif method == 'SVM_linear':
            model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
        elif method == 'SVM_rbf':
            model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        
        model.fit(X_scaled, y)
        
        # Save model with metadata
        model_data = {
            'model': model,
            'scaler': scaler,
            'selected_indices': bio_indices,
            'selected_mask': selected_mask if selected_mask is not None else np.ones(X_bio.shape[1], dtype=bool),
            'frequencies': freq,
            'selected_frequencies': selected_freq,
            'timepoint': timepoint,
            'method': method,
            'feature_selector': {
                'biological': biological_selector.get_config(),
                'lasso': lasso_selector.get_config()
            }
        }
        
        joblib.dump(model_data, model_path)
        
    except Exception as e:
        print(f"Warning: Could not save model: {e}")

def save_timepoint_results(results, timepoint, output_dir):
    """Save results for a timepoint"""
    # Save summary
    summary_data = []
    for method, metrics in results.items():
        summary_data.append({
            'Method': method,
            'Mean_Accuracy': metrics['mean_accuracy'],
            'Std_Accuracy': metrics['std_accuracy'],
            'CI_Lower': metrics['confidence_interval'][0],
            'CI_Upper': metrics['confidence_interval'][1],
            'N_Splits': metrics['n_splits'],
            'Avg_Preselected_Features': np.mean(metrics['preselected_features']),
            'Avg_Final_Features': np.mean(metrics['final_features']),
            'Train_Size_Distribution': str(metrics['train_size_distribution'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, f'summary_{timepoint}.csv'), index=False)
    
    # Save detailed accuracies
    for method, metrics in results.items():
        acc_df = pd.DataFrame({
            'split': range(len(metrics['all_accuracies'])),
            'accuracy': metrics['all_accuracies'],
            'train_size': metrics['train_sizes'],
            'preselected_features': metrics['preselected_features'],
            'final_features': metrics['final_features']
        })
        acc_df.to_csv(os.path.join(output_dir, f'{method}_accuracies_{timepoint}.csv'), index=False)

def create_summary_plot(all_results, output_dir, verbose=True):
    """Create summary plot of all results"""
    # Prepare data
    plot_data = []
    for timepoint, methods in all_results.items():
        for method, metrics in methods.items():
            plot_data.append({
                'Timepoint': timepoint,
                'Method': method,
                'Accuracy': metrics['mean_accuracy'],
                'CI_Lower': metrics['confidence_interval'][0],
                'CI_Upper': metrics['confidence_interval'][1],
                'Final_Features': np.mean(metrics['final_features'])
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    timepoints_ordered = ['0min', '20min', '30min', '60min']
    methods = ['LDA', 'PLS_DA', 'SVM_linear', 'SVM_rbf']
    
    # Plot 1: Accuracy
    n_timepoints = len(timepoints_ordered)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df['Method'] == method]
        
        accuracies, ci_lowers, ci_uppers = [], [], []
        for tp in timepoints_ordered:
            tp_data = method_data[method_data['Timepoint'] == tp]
            if len(tp_data) > 0:
                accuracies.append(tp_data['Accuracy'].values[0])
                ci_lowers.append(tp_data['CI_Lower'].values[0])
                ci_uppers.append(tp_data['CI_Upper'].values[0])
            else:
                accuracies.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)
        
        x_pos = np.arange(n_timepoints) + i * bar_width
        
        for j, tp in enumerate(timepoints_ordered):
            ax1.bar(x_pos[j], accuracies[j], width=bar_width,
                   color='lightgrey' if i % 2 == 0 else 'darkgrey',
                   edgecolor='black', alpha=0.9)
            
            errors_lower = accuracies[j] - ci_lowers[j]
            errors_upper = ci_uppers[j] - accuracies[j]
            
            ax1.errorbar(x_pos[j], accuracies[j],
                       yerr=[[errors_lower], [errors_upper]],
                       fmt='none', c='black', capsize=5)
            
            ax1.text(x_pos[j], accuracies[j] + 0.02,
                   f'{accuracies[j]:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_xlabel('Time Group', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(np.arange(n_timepoints) + (n_methods - 1) * bar_width / 2)
    ax1.set_xticklabels(timepoints_ordered)
    ax1.set_title('Accuracy with 95% Confidence Intervals', fontsize=16)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Features
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df['Method'] == method]
        
        features = []
        for tp in timepoints_ordered:
            tp_data = method_data[method_data['Timepoint'] == tp]
            features.append(tp_data['Final_Features'].values[0] if len(tp_data) > 0 else 0)
        
        x_pos = np.arange(n_timepoints) + i * bar_width
        ax2.bar(x_pos, features, width=bar_width,
               color='lightgrey' if i % 2 == 0 else 'darkgrey',
               edgecolor='black', alpha=0.9)
        
        for j, feat in enumerate(features):
            ax2.text(x_pos[j], feat + 0.5, f'{feat:.0f}', 
                   ha='center', va='bottom', fontsize=10)
    
    ax2.set_xlabel('Time Group', fontsize=14)
    ax2.set_ylabel('Features Selected', fontsize=14)
    ax2.set_xticks(np.arange(n_timepoints) + (n_methods - 1) * bar_width / 2)
    ax2.set_xticklabels(timepoints_ordered)
    ax2.set_title('Features Selected by LASSO', fontsize=16)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MRSA/MSSA Classification Analysis\n(Biological Pre-selection + LASSO Feature Selection)', 
                fontsize=18, y=1.02)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'analysis_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if verbose:
        plt.show()
    else:
        plt.close()
    
    return plot_path
