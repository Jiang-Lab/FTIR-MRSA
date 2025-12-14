import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class BiologicalFeatureSelector:
    """Biological feature selection based on spectral regions"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    def get_biologically_relevant_regions(self, timepoint):
        """Get spectral regions for timepoint"""
        regions = {
            'all': [(950, 1100), (1125, 1170), (1300, 1400), 
                   (1500, 1530), (1560, 1685), (1725, 1745)],
            '0min': [(950, 1100), (1125, 1170), (1300, 1400),
                    (1500, 1530), (1560, 1685), (1725, 1745)],
            '20min': [(950, 1100), (1125, 1170), (1300, 1400),
                     (1500, 1530), (1560, 1685), (1725, 1745)],
            '30min': [(950, 1100), (1125, 1170), (1500, 1530),
                     (1560, 1685), (1725, 1745)],
            '60min': [(950, 1100), (1125, 1170), (1300, 1400),
                     (1500, 1530), (1560, 1685), (1725, 1745)]
        }
        return regions.get(timepoint, regions['all'])
    
    def pre_select_features(self, X, freq, timepoint, verbose=None):
        """Pre-select features based on biological regions"""
        if verbose is None:
            verbose = self.verbose
        
        freq_array = np.array(freq)
        regions = self.get_biologically_relevant_regions(timepoint)
        
        selected_indices = []
        for low, high in regions:
            mask = (freq_array >= low) & (freq_array <= high)
            selected_indices.extend(np.where(mask)[0])
        
        selected_indices = sorted(set(selected_indices))
        
        if verbose:
            print(f"Pre-selected {len(selected_indices)} features from {len(freq)} total")
        
        return X[:, selected_indices], selected_indices
    
    def get_config(self):
        """Get selector configuration"""
        return {'type': 'biological', 'verbose': self.verbose}

class LassoFeatureSelector:
    """LASSO feature selection with bootstrap stability"""
    
    def __init__(self, n_bootstrap=100, stability_threshold=0.2, verbose=True):
        self.n_bootstrap = n_bootstrap
        self.stability_threshold = stability_threshold
        self.verbose = verbose
    
    def optimize_wavelengths(self, X_train, y_train, n_bootstrap=None, 
                            stability_threshold=None, verbose=None):
        """Select features using LASSO with bootstrap stability"""
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
        if stability_threshold is None:
            stability_threshold = self.stability_threshold
        if verbose is None:
            verbose = self.verbose
        
        if len(X_train) == 0 or len(np.unique(y_train)) < 2:
            return None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        y_numeric = np.where(y_train == 'MRSA', 1, 0)
        
        feature_stability = np.zeros(X_scaled.shape[1])
        
        for i in range(n_bootstrap):
            try:
                X_resampled, y_resampled, y_numeric_resampled = resample(
                    X_scaled, y_train, y_numeric, replace=True,
                    n_samples=len(X_scaled), random_state=i
                )
                
                if len(np.unique(y_resampled)) < 2:
                    continue
                
                n_folds = min(10, len(X_resampled))
                lasso = LassoCV(cv=n_folds, random_state=42+i, max_iter=5000, tol=1e-4)
                lasso.fit(X_resampled, y_numeric_resampled)
                
                selected_features = lasso.coef_ != 0
                feature_stability += selected_features.astype(int)
                    
            except Exception:
                continue
        
        stability_scores = feature_stability / n_bootstrap
        selected_mask = stability_scores >= stability_threshold
        
        if verbose:
            print(f"LASSO selected {np.sum(selected_mask)} features "
                  f"with stability ≥ {stability_threshold}")
        
        return selected_mask
    
    def get_config(self):
        """Get selector configuration"""
        return {
            'type': 'lasso',
            'n_bootstrap': self.n_bootstrap,
            'stability_threshold': self.stability_threshold,
            'verbose': self.verbose
        }

class StrainAwareSplitter:
    """Generate strain-preserving train/test splits"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
    
    @staticmethod
    def get_strain_info(target_label):
        """Extract strain information from label"""
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
    
    def generate_flexible_splits(self, X, y, target_labels, 
                                min_train_samples=12, max_train_samples=15,
                                n_splits_per_size=50, verbose=None):
        """Generate strain-preserving splits"""
        if verbose is None:
            verbose = self.verbose
        
        strains = [self.get_strain_info(label) for label in target_labels]
        
        strain_groups = {}
        for i, strain in enumerate(strains):
            if strain not in strain_groups:
                strain_groups[strain] = []
            strain_groups[strain].append(i)
        
        if verbose:
            print(f"Strain groups: {[(s, len(indices)) for s, indices in strain_groups.items()]}")
        
        main_strains = [s for s in ['MRSA_JE2', 'MRSA_43300', 'MSSA_6835', 'MSSA_RN'] 
                       if s in strain_groups and strain_groups[s]]
        
        if len(main_strains) < 4:
            if verbose:
                print(f"Warning: Only {len(main_strains)} main strains found")
            main_strains = [s for s in strain_groups.keys() if s != 'UNKNOWN']
        
        valid_splits = []
        total_samples = len(X)
        
        if verbose:
            print(f"Total samples: {total_samples}")
            print(f"Train sizes: {min_train_samples} to {max_train_samples}")
        
        for train_size in range(min_train_samples, max_train_samples + 1):
            test_size = total_samples - train_size
            
            if test_size < 1:
                if verbose:
                    print(f"  Skipping train_size={train_size} (test_size={test_size})")
                continue
            
            successful_for_size = 0
            
            for _ in range(n_splits_per_size):
                all_indices = set(range(total_samples))
                train_indices = set()
                
                for strain in main_strains:
                    if strain in strain_groups:
                        available = [i for i in strain_groups[strain] if i in all_indices]
                        if available:
                            selected = np.random.choice(available, size=1, replace=False)
                            train_indices.add(selected[0])
                            all_indices.remove(selected[0])
                
                remaining = train_size - len(train_indices)
                if remaining > 0 and len(all_indices) >= remaining:
                    additional = np.random.choice(list(all_indices), size=remaining, replace=False)
                    train_indices.update(additional)
                    for idx in additional:
                        all_indices.remove(idx)
                
                train_indices = list(train_indices)
                test_indices = list(all_indices)
                
                if (len(train_indices) == train_size and len(test_indices) > 0 and
                    len(np.unique([y[i] for i in train_indices])) >= 2):
                    valid_splits.append((train_indices, test_indices))
                    successful_for_size += 1
            
            if verbose:
                print(f"  Train_size={train_size}: {successful_for_size} splits")
        
        if verbose:
            print(f"Generated {len(valid_splits)} total splits")
        
        return valid_splits
