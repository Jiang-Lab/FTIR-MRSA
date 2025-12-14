import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import os
import warnings
import itertools
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set style for better plots with LARGE fonts
plt.style.use('default')
sns.set_palette("viridis")
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

def get_biologically_relevant_regions(timepoint):
    """
    Define wavelength regions based on biological significance from your paper
    """
    # Broad regions that include the specific wavenumbers mentioned in your paper
    regions = {
        'all': [
            (950, 1100),   # Cell wall stress region (1025 cm⁻¹)
            (1125, 1170),   # Natural differences (1158 cm⁻¹)
            (1300, 1400),
            (1500, 1530),   # Peptidoglycan precursors (1516, 1515, 1509 cm⁻¹)
            (1560, 1685),   # Amide I
            (1725, 1745),   # Lipid region - membrane damage (1735 cm⁻¹)
        ],
        '0min': [
            (950, 1100),   # Cell wall stress region (1025 cm⁻¹)
            (1125, 1170),   # Natural differences (1158 cm⁻¹)
            (1300, 1400),
            (1500, 1530),   # Peptidoglycan precursors (1516, 1515, 1509 cm⁻¹)
            (1560, 1685),   # Amide I
            (1725, 1745),   # Lipid region - membrane damage (1735 cm⁻¹)
        ],
        '20min': [
            (950, 1100),   # Cell wall stress region (1025 cm⁻¹)
            (1125, 1170),   # Natural differences (1158 cm⁻¹)
            (1300, 1400),
            (1500, 1530),   # Peptidoglycan precursors (1516, 1515, 1509 cm⁻¹)
            (1560, 1685),   # Amide I
            (1725, 1745),   # Lipid region - membrane damage (1735 cm⁻¹)
        ],
        '30min': [
            (950, 1100),   # Cell wall stress region (1025 cm⁻¹)
            (1125, 1170),   # Natural differences (1158 cm⁻¹)
            (1500, 1530),   # Peptidoglycan precursors (1516, 1515, 1509 cm⁻¹)
            (1560, 1685),   # Amide I
            (1725, 1745),   # Lipid region - membrane damage (1735 cm⁻¹)
        ],
        '60min': [
            (950, 1100),   # Cell wall stress region (1025 cm⁻¹)
            (1125, 1170),   # Natural differences (1158 cm⁻¹)
            (1300, 1400),
            (1500, 1530),   # Peptidoglycan precursors (1516, 1515, 1509 cm⁻¹)
            (1560, 1685),   # Amide I
            (1725, 1745),   # Lipid region - membrane damage (1735 cm⁻¹)
        ]
    }
    
    return regions.get(timepoint, regions['all'])

def pre_select_biologically_relevant_features(X, freq, timepoint):
    """
    Pre-select wavelengths based on biologically relevant regions
    """
    # Convert freq to numpy array for proper comparison - THIS FIXES THE ERROR
    freq_array = np.array(freq)
    
    regions = get_biologically_relevant_regions(timepoint)
    
    selected_indices = []
    for low, high in regions:
        mask = (freq_array >= low) & (freq_array <= high)
        selected_indices.extend(np.where(mask)[0])
    
    # Remove duplicates and sort
    selected_indices = sorted(set(selected_indices))
    
    print(f"Pre-selected {len(selected_indices)} features from {len(freq)} total for {timepoint}")
    print(f"Selected regions: {regions}")
    
    return X[:, selected_indices], selected_indices

def optimize_wavelengths_with_lasso_loocv(X_train, y_train, freq, n_bootstrap=100, stability_threshold=0.2):
    """
    Find optimal wavelengths using LASSO with increased bootstrap and stability threshold
    """
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return None
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Convert labels to numeric for LASSO
    y_numeric = np.where(y_train == 'MRSA', 1, 0)
    
    # Track feature selection stability across bootstraps
    feature_stability = np.zeros(X_scaled.shape[1])
    
    for i in range(n_bootstrap):
        try:
            # Bootstrap resample
            X_resampled, y_resampled, y_numeric_resampled = resample(
                X_scaled, y_train, y_numeric, replace=True, 
                n_samples=len(X_scaled), random_state=i
            )
            
            # Ensure we have both classes
            unique_classes = np.unique(y_resampled)
            if len(unique_classes) < 2:
                continue
            
            # LASSO with cross-validation to find optimal alpha
            n_folds = min(10, len(X_resampled))
            lasso = LassoCV(cv=n_folds, random_state=42+i, max_iter=5000, tol=1e-4)
            lasso.fit(X_resampled, y_numeric_resampled)
            
            # Track selected features (non-zero coefficients)
            selected_features = lasso.coef_ != 0
            feature_stability += selected_features.astype(int)
                
        except Exception as e:
            continue
    
    # Calculate feature stability scores
    stability_scores = feature_stability / n_bootstrap
    
    # Select features based on stability threshold
    selected_mask = stability_scores >= stability_threshold
    
    print(f"LASSO selected {np.sum(selected_mask)} features with stability ≥ {stability_threshold}")
    print(f"Stability scores range: {stability_scores.min():.3f} - {stability_scores.max():.3f}")
    
    return selected_mask

def get_strain_info(target_label):
    """
    Extract strain information from target label
    Based on your actual target labels
    """
    label_str = str(target_label).upper()
    
    if 'JE2' in label_str:
        return 'MRSA_JE2'
    elif '43300' in label_str:
        return 'MRSA_43300'
    elif '6835' in label_str:
        return 'MSSA_6835'
    elif 'RN' in label_str:
        return 'MSSA_RN'
    elif 'MRSA' in label_str and 'MSSA' not in label_str:
        return 'MRSA_UNKNOWN'
    elif 'MSSA' in label_str and 'MRSA' not in label_str:
        return 'MSSA_UNKNOWN'
    else:
        return 'UNKNOWN'

def generate_flexible_strain_splits(X, y, target_labels, min_train_samples=12, max_train_samples=15):
    """
    Generate flexible train/test splits with varying training sizes (12-15 samples)
    while preserving strain diversity
    """
    # Get strain information for each sample
    strains = [get_strain_info(label) for label in target_labels]
    
    # Group samples by strain
    strain_groups = {}
    for i, strain in enumerate(strains):
        if strain not in strain_groups:
            strain_groups[strain] = []
        strain_groups[strain].append(i)
    
    print(f"Strain groups: {[(strain, len(indices)) for strain, indices in strain_groups.items()]}")
    
    # Identify the main strains present
    main_strains = [strain for strain in ['MRSA_JE2', 'MRSA_43300', 'MSSA_6835', 'MSSA_RN'] 
                   if strain in strain_groups and len(strain_groups[strain]) > 0]
    
    # If we don't have all 4 main strains, adjust strategy
    if len(main_strains) < 4:
        print(f"Warning: Only found {len(main_strains)} main strains: {main_strains}")
        # Use all available strains
        main_strains = [strain for strain in strain_groups.keys() if strain != 'UNKNOWN']
    
    valid_splits = []
    total_samples = len(X)
    
    print(f"Total samples available: {total_samples}")
    print(f"Generating splits with training sizes from {min_train_samples} to {max_train_samples}")
    
    # Generate splits with different training sizes
    for train_size in range(min_train_samples, max_train_samples + 1):
        test_size = total_samples - train_size
        
        # We need at least 1 sample in test
        if test_size < 1:
            print(f"  Skipping train_size={train_size}: test_size would be {test_size} (need at least 1)")
            continue
            
        print(f"  Generating splits with train_size={train_size}, test_size={test_size}")
        
        # Generate multiple random splits for each training size
        n_splits_per_size = 20  # Reduced for faster testing
        
        print(f"    Trying {n_splits_per_size} splits per training size")
        
        successful_splits_for_size = 0
        
        for split_idx in range(n_splits_per_size):
            # Start with all indices as potential test samples
            all_indices = set(range(total_samples))
            train_indices = set()
            
            # First: ensure we get at least 1 sample from each main strain in training
            for strain in main_strains:
                if strain in strain_groups and strain_groups[strain]:
                    available_from_strain = [i for i in strain_groups[strain] if i in all_indices]
                    if available_from_strain:
                        selected = np.random.choice(available_from_strain, size=1, replace=False)
                        train_indices.add(selected[0])
                        all_indices.remove(selected[0])
            
            # If we still need more training samples, randomly select from remaining
            remaining_needed = train_size - len(train_indices)
            if remaining_needed > 0 and len(all_indices) >= remaining_needed:
                additional_train = np.random.choice(list(all_indices), size=remaining_needed, replace=False)
                train_indices.update(additional_train)
                for idx in additional_train:
                    all_indices.remove(idx)
            
            train_indices = list(train_indices)
            test_indices = list(all_indices)
            
            # Only use this split if we have the desired training size and both classes
            if len(train_indices) == train_size and len(test_indices) > 0:
                train_classes = [y[i] for i in train_indices]
                test_classes = [y[i] for i in test_indices]
                
                # Check we have both classes in training and at least one in test
                if (len(np.unique(train_classes)) >= 2 and len(np.unique(test_classes)) >= 1):
                    valid_splits.append((train_indices, test_indices))
                    successful_splits_for_size += 1
        
        print(f"    Generated {successful_splits_for_size} valid splits for train_size={train_size}")
    
    print(f"Generated {len(valid_splits)} total valid flexible splits")
    
    # Analyze the training size distribution
    if valid_splits:
        train_sizes = [len(train_idx) for train_idx, _ in valid_splits]
        size_counts = Counter(train_sizes)
        print(f"Training size distribution: {dict(size_counts)}")
    
    return valid_splits

def process_single_split_improved(args):
    """
    Improved version with biological pre-selection and better LASSO
    """
    train_idx, test_idx, X, y, freq, method, timepoint = args
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    try:
        # STEP 1: Pre-select biologically relevant features
        X_train_preselected, selected_indices = pre_select_biologically_relevant_features(X_train, freq, timepoint)
        X_test_preselected = X_test[:, selected_indices]
        freq_preselected = [freq[i] for i in selected_indices]
        
        # STEP 2: Apply LASSO feature selection on pre-selected features
        selected_mask = optimize_wavelengths_with_lasso_loocv(
            X_train_preselected, y_train, freq_preselected, 
            n_bootstrap=100,  # Increased
            stability_threshold=0.1  # Increased
        )
        
        if selected_mask is None or np.sum(selected_mask) == 0:
            # Fallback: use all pre-selected features
            X_train_selected = X_train_preselected
            X_test_selected = X_test_preselected
            n_features = X_train_preselected.shape[1]
            print("  LASSO selected no features, using all pre-selected features")
        else:
            X_train_selected = X_train_preselected[:, selected_mask]
            X_test_selected = X_test_preselected[:, selected_mask]
            n_features = np.sum(selected_mask)
            print(f"  LASSO selected {n_features} features from {X_train_preselected.shape[1]} pre-selected")
        
        # Standardize selected features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train and predict with simple models (no hyperparameter optimization)
        if method == 'LDA':
            model = LinearDiscriminantAnalysis()
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)
        elif method == 'PLS_DA':
            y_train_numeric = np.where(y_train == 'MRSA', 1, 0)
            # Use fixed number of components
            n_components = min(3, X_train_scaled.shape[1], len(X_train_scaled) - 1)
            model = PLSRegression(n_components=n_components)
            model.fit(X_train_scaled, y_train_numeric)
            y_pred_proba = model.predict(X_test_scaled).flatten()
            pred = np.where(y_pred_proba > 0.5, 'MRSA', 'MSSA')
            proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        elif method == 'SVM_linear':
            model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)
        elif method == 'SVM_rbf':
            model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            proba = model.predict_proba(X_test_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate accuracy for this split
        accuracy = accuracy_score(y_test, pred)
        
        return {
            'test_indices': test_idx,
            'predictions': list(pred),
            'true_labels': list(y_test),
            'probabilities': proba.tolist(),
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'preselected_features': X_train_preselected.shape[1],
            'final_features': n_features,
            'success': True
        }
    
    except Exception as e:
        print(f"Error in split processing: {e}")
        return {
            'test_indices': test_idx,
            'predictions': [],
            'true_labels': list(y_test),
            'probabilities': [],
            'accuracy': 0.0,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'success': False,
            'error': str(e)
        }

def run_improved_analysis():
    """
    Main function with biological pre-selection and improved LASSO
    """
    print("IMPROVED STRAIN-PRESERVING ANALYSIS WITH BIOLOGICAL FEATURE SELECTION")
    print("=" * 80)
    print("Strategy: Biological pre-selection → LASSO feature selection → Simple models")
    print("=" * 80)
    
    # Load your data
    df = pd.read_excel('./processed_spectra_normalized_2nd_derivative_balenced.ods', 
                       index_col=0, engine='odf')
    freq = list(df.columns[0:490].values)
    
    df['simple_target'] = df['target'].apply(lambda x: 'MRSA' if 'MRSA' in x else 'MSSA')
    df['time_info'] = df['target'].apply(lambda x: re.sub(r'(MRSA|MSSA)\s*', '', x).strip())
    
    timepoints = ['0min', '20min', '30min', '60min']
    time_groups = {
        '0min': ['0 min'],
        '20min': ['20 min'], 
        '30min': ['30 min'],
        '60min': ['60 min']
    }
    
    methods = ['LDA', 'PLS_DA', 'SVM_linear', 'SVM_rbf']  # ALL methods from original
    
    all_timepoint_results = {}
    
    for timepoint in timepoints:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {timepoint}")
        print(f"{'='*60}")
        
        # Filter data for this time point
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
            print(f"  ✗ No data available for {timepoint}")
            continue
        
        # Filter out control samples
        non_control_mask = ~df_time['target'].str.contains('control', case=False, na=False)
        df_time_eval = df_time[non_control_mask]
        
        total_samples = len(df_time_eval)
        print(f"  Total non-control samples: {total_samples}")
        
        if total_samples < 13:  # Need at least 13 samples (12 train + 1 test)
            print(f"  ✗ Not enough non-control samples for {timepoint}. Need at least 13, have {total_samples}")
            continue
        
        # Prepare features and labels
        X = df_time_eval[freq].values
        y = df_time_eval['simple_target'].values
        target_labels = df_time_eval['target'].values
        
        print(f"  ✓ Samples: {len(X)}")
        print(f"  ✓ Class distribution: {np.unique(y, return_counts=True)}")
        
        # Generate all valid splits
        splits = generate_flexible_strain_splits(X, y, target_labels, 
                                               min_train_samples=12, 
                                               max_train_samples=15)
        
        if not splits:
            print("  ✗ No valid splits generated!")
            continue
        
        # Process splits
        timepoint_results = {}
        
        for method in methods:
            print(f"\n  Running {method} with biological pre-selection + LASSO...")
            
            # Prepare arguments for multiprocessing
            n_workers = min(mp.cpu_count(), 16)
            args_list = [(train_idx, test_idx, X, y, freq, method, timepoint) 
                        for train_idx, test_idx in splits[:80]]  # Limit to first 50 splits for speed
            
            # Process splits
            all_results = []
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(process_single_split_improved, arg) for arg in args_list]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {method}"):
                    try:
                        result = future.result(timeout=120)  # 2 minute timeout
                        all_results.append(result)
                    except Exception as e:
                        print(f"Task failed: {e}")
                        all_results.append({'success': False, 'error': str(e)})
            
            # Aggregate results
            y_true_all, y_pred_all = [], []
            accuracies = []
            train_sizes = []
            preselected_features = []
            final_features = []
            successful_runs = 0
            
            for result in all_results:
                if result['success']:
                    y_true_all.extend(result['true_labels'])
                    y_pred_all.extend(result['predictions'])
                    accuracies.append(result['accuracy'])
                    train_sizes.append(result['train_size'])
                    preselected_features.append(result['preselected_features'])
                    final_features.append(result['final_features'])
                    successful_runs += 1
            
            if not accuracies:
                print(f"    ✗ No successful runs for {method}")
                continue
            
            print(f"    ✓ Successful runs: {successful_runs}/{len(all_results)}")
            
            # Calculate overall metrics
            overall_accuracy = accuracy_score(y_true_all, y_pred_all)
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            
            # Calculate confidence intervals (95%)
            n_splits = len(accuracies)
            if n_splits > 1:
                standard_error = std_accuracy / np.sqrt(n_splits)
                confidence_interval = (
                    max(0.0, mean_accuracy - 1.96 * standard_error),
                    min(1.0, mean_accuracy + 1.96 * standard_error)
                )
                ci_size = confidence_interval[1] - confidence_interval[0]
            else:
                confidence_interval = (mean_accuracy, mean_accuracy)
                ci_size = 0.0
            
            # Analyze training size distribution
            size_counts = Counter(train_sizes)
            
            print(f"    ✓ Mean accuracy: {mean_accuracy:.3f}")
            print(f"    ✓ 95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]")
            print(f"    ✓ Avg pre-selected features: {np.mean(preselected_features):.1f}")
            print(f"    ✓ Avg final features: {np.mean(final_features):.1f}")
            print(f"    ✓ Training size distribution: {dict(size_counts)}")
            
            timepoint_results[method] = {
                'overall_accuracy': overall_accuracy,
                'mean_accuracy': mean_accuracy,
                'std_accuracy': std_accuracy,
                'standard_error': standard_error,
                'confidence_interval': confidence_interval,
                'ci_size': ci_size,
                'n_splits': n_splits,
                'all_accuracies': accuracies,
                'train_sizes': train_sizes,
                'preselected_features': preselected_features,
                'final_features': final_features,
                'train_size_distribution': dict(size_counts),
                'y_true': y_true_all,
                'y_pred': y_pred_all
            }
        
        if timepoint_results:
            all_timepoint_results[timepoint] = timepoint_results
            save_improved_results(timepoint_results, timepoint)
        else:
            print(f"  ✗ No successful results for {timepoint}")
    
    # Create plots
    if all_timepoint_results:
        plot_improved_results(all_timepoint_results)
        print(f"\nAnalysis complete! Used biological pre-selection + improved LASSO")
        print("Results saved in 'improved_results' directory")
    else:
        print("No results generated for any timepoint!")
    
    return all_timepoint_results

def save_improved_results(timepoint_results, timepoint):
    """Save results from improved analysis"""
    os.makedirs('improved_results', exist_ok=True)
    
    # Save summary statistics
    summary_data = []
    for method, results in timepoint_results.items():
        summary_data.append({
            'Method': method,
            'Overall_Accuracy': results['overall_accuracy'],
            'Mean_Accuracy': results['mean_accuracy'],
            'Std_Accuracy': results['std_accuracy'],
            'Standard_Error': results['standard_error'],
            'CI_Lower': results['confidence_interval'][0],
            'CI_Upper': results['confidence_interval'][1],
            'CI_Size': results['ci_size'],
            'N_Splits': results['n_splits'],
            'Avg_Preselected_Features': np.mean(results['preselected_features']),
            'Avg_Final_Features': np.mean(results['final_features']),
            'Train_Size_Distribution': str(results['train_size_distribution'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'improved_results/summary_{timepoint}.csv', index=False)
    
    # Save detailed accuracy values for each split
    for method, results in timepoint_results.items():
        accuracies_df = pd.DataFrame({
            'split': range(len(results['all_accuracies'])),
            'accuracy': results['all_accuracies'],
            'train_size': results['train_sizes'],
            'preselected_features': results['preselected_features'],
            'final_features': results['final_features']
        })
        accuracies_df.to_csv(f'improved_results/{method}_accuracies_{timepoint}.csv', index=False)

def plot_improved_results(all_timepoint_results):
    """Plot results from improved analysis"""
    # Prepare data for plotting
    plot_data = []
    
    for timepoint, methods_dict in all_timepoint_results.items():
        for method, results in methods_dict.items():
            plot_data.append({
                'Timepoint': timepoint,
                'Method': method,
                'Accuracy': results['mean_accuracy'],
                'CI_Lower': results['confidence_interval'][0],
                'CI_Upper': results['confidence_interval'][1],
                'CI_Size': results['ci_size'],
                'N_Splits': results['n_splits'],
                'Final_Features': np.mean(results['final_features'])
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Accuracy with confidence intervals
    timepoints_ordered = ['0min', '20min', '30min', '60min']
    methods = ['LDA', 'PLS_DA', 'SVM_linear', 'SVM_rbf']
    
    # Define colors for timepoints
    timepoint_colors = {
        '0min': 'lightgrey',
        '20min': 'darkgrey', 
        '30min': 'grey',
        '60min': 'dimgrey'
    }
    
    # Bar positions
    n_timepoints = len(timepoints_ordered)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods
    
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df['Method'] == method]
        
        accuracies = []
        ci_lowers = []
        ci_uppers = []
        
        for timepoint in timepoints_ordered:
            time_data = method_data[method_data['Timepoint'] == timepoint]
            if len(time_data) > 0:
                accuracies.append(time_data['Accuracy'].values[0])
                ci_lowers.append(time_data['CI_Lower'].values[0])
                ci_uppers.append(time_data['CI_Upper'].values[0])
            else:
                accuracies.append(0)
                ci_lowers.append(0)
                ci_uppers.append(0)
        
        x_pos = np.arange(n_timepoints) + i * bar_width
        
        for j, timepoint in enumerate(timepoints_ordered):
            ax1.bar(x_pos[j], accuracies[j], width=bar_width,
                   color=timepoint_colors[timepoint],
                   edgecolor='black', linewidth=2, alpha=0.9)
            
            errors_lower = accuracies[j] - ci_lowers[j]
            errors_upper = ci_uppers[j] - accuracies[j]
            
            ax1.errorbar(x_pos[j], accuracies[j],
                       yerr=[[errors_lower], [errors_upper]],
                       fmt='none', c='black', capsize=6, capthick=2,
                       elinewidth=2)
            
            # Add accuracy value
            ax1.text(x_pos[j], 1.02, f'{accuracies[j]:.3f}', 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax1.set_xlabel('Time Group', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=18, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(np.arange(n_timepoints) + (n_methods - 1) * bar_width / 2)
    ax1.set_xticklabels(timepoints_ordered, fontsize=16, fontweight='bold')
    ax1.set_title('Accuracy with 95% Confidence Intervals\n(Biological Pre-selection + LASSO)', 
                 fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Number of features used
    for i, method in enumerate(methods):
        method_data = plot_df[plot_df['Method'] == method]
        
        features = []
        for timepoint in timepoints_ordered:
            time_data = method_data[method_data['Timepoint'] == timepoint]
            if len(time_data) > 0:
                features.append(time_data['Final_Features'].values[0])
            else:
                features.append(0)
        
        x_pos = np.arange(n_timepoints) + i * bar_width
        ax2.bar(x_pos, features, width=bar_width,
               color=[timepoint_colors[tp] for tp in timepoints_ordered],
               edgecolor='black', linewidth=2, alpha=0.9)
        
        # Add feature count values
        for j, feat in enumerate(features):
            ax2.text(x_pos[j], feat + 0.5, f'{feat:.0f}', 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax2.set_xlabel('Time Group', fontsize=18, fontweight='bold')
    ax2.set_ylabel('Number of Features Selected', fontsize=18, fontweight='bold')
    ax2.set_xticks(np.arange(n_timepoints) + (n_methods - 1) * bar_width / 2)
    ax2.set_xticklabels(timepoints_ordered, fontsize=16, fontweight='bold')
    ax2.set_title('Number of Features Selected by LASSO', fontsize=20, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('improved_results/improved_analysis_results.png', bbox_inches='tight', dpi=300)
    plt.savefig('improved_results/improved_analysis_results.pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_improved_analysis()

