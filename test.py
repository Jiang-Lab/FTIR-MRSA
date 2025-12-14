import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
import os

print("MINIMAL MRSA ANALYSIS")
print("="*60)

# Create results directory
os.makedirs("minimal_results", exist_ok=True)

# Load data
try:
    df = pd.read_excel('Data/processed_spectra_normalized_2nd_derivative_balenced.ods', 
                       index_col=0, engine='odf')
    print(f"✓ Data loaded: {df.shape}")
    
    # Check target column
    if 'target' not in df.columns:
        print("✗ 'target' column not found")
        exit()
    
    # Simple MRSA/MSSA classification
    df['binary_label'] = df['target'].apply(
        lambda x: 'MRSA' if 'MRSA' in str(x).upper() else 'MSSA' if 'MSSA' in str(x).upper() else None
    )
    
    # Remove samples without clear label
    df = df[df['binary_label'].notna()]
    
    print(f"✓ Samples with clear labels: {len(df)}")
    print(f"  MRSA: {(df['binary_label'] == 'MRSA').sum()}")
    print(f"  MSSA: {(df['binary_label'] == 'MSSA').sum()}")
    
    # Get spectral columns (first 490 columns)
    spectral_cols = []
    for i, col in enumerate(df.columns):
        if col not in ['target', 'binary_label'] and i < 490:
            spectral_cols.append(col)
    
    print(f"✓ Spectral features: {len(spectral_cols)}")
    
    # Prepare data
    X = df[spectral_cols].values
    y = df['binary_label'].values
    
    # Simple LOO-CV
    loo = LeaveOneOut()
    accuracies = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracies.append(accuracy_score([y_test], [y_pred]))
    
    print(f"\n✓ Results:")
    print(f"  Mean Accuracy: {np.mean(accuracies):.3f}")
    print(f"  Std Deviation: {np.std(accuracies):.3f}")
    print(f"  Range: {min(accuracies):.3f} - {max(accuracies):.3f}")
    
    # Save results
    results_df = pd.DataFrame({
        'split': range(len(accuracies)),
        'accuracy': accuracies
    })
    results_df.to_csv('minimal_results/accuracy_results.csv', index=False)
    
    print(f"\n✓ Results saved to minimal_results/accuracy_results.csv")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
