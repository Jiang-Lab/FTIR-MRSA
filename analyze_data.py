import argparse
import os
import sys
import time
import warnings
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Analyze MRSA/MSSA spectra with biological feature selection')
    parser.add_argument('--data', '-d', default='data/processed_spectra_normalized_2nd_derivative_balenced.ods',
                       help='Path to data file (ODS/Excel)')
    parser.add_argument('--output', '-o', default='results',
                       help='Output directory for results')
    parser.add_argument('--timepoints', '-t', nargs='+', 
                       default=['0min', '20min', '30min', '60min'],
                       help='Timepoints to analyze')
    parser.add_argument('--methods', '-m', nargs='+',
                       default=['LDA', 'PLS_DA', 'SVM_linear', 'SVM_rbf'],
                       help='Classification methods to use')
    parser.add_argument('--n-splits', type=int, default=50,
                       help='Number of splits to run per method')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick test with minimal splits (5 per method)')
    
    args = parser.parse_args()
    
    # Quick mode override
    if args.quick:
        args.n_splits = 5
        args.timepoints = ['0min', '30min']  # Just test 2 timepoints
        print("Running in QUICK TEST mode (5 splits, 2 timepoints)")
    
    # Set n_jobs
    if args.n_jobs == -1:
        args.n_jobs = mp.cpu_count()
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found at {args.data}")
        print("Please provide a valid data file with --data argument")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Import and run analysis
    try:
        from src.analysis.pipeline import run_full_analysis
        
        print("="*70)
        print("MRSA/MSSA CLASSIFICATION ANALYSIS")
        print("="*70)
        print(f"Data: {args.data}")
        print(f"Timepoints: {', '.join(args.timepoints)}")
        print(f"Methods: {', '.join(args.methods)}")
        print(f"Splits per method: {args.n_splits}")
        print(f"Parallel jobs: {args.n_jobs}")
        print(f"Output: {args.output}")
        print(f"Quick mode: {'Yes' if args.quick else 'No'}")
        print("="*70)
        
        start_time = time.time()
        
        results = run_full_analysis(
            data_path=args.data,
            timepoints=args.timepoints,
            methods=args.methods,
            output_dir=args.output,
            n_splits=args.n_splits,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
            quick_mode=args.quick
        )
        
        elapsed_time = time.time() - start_time
        
        if results:
            print("\n" + "="*70)
            print("ANALYSIS COMPLETE!")
            print("="*70)
            print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
            print(f"Results saved in '{args.output}' directory:")
            
            # List generated files
            for root, dirs, files in os.walk(args.output):
                for file in files:
                    if file.endswith(('.csv', '.png', '.txt')):
                        rel_path = os.path.relpath(os.path.join(root, file), args.output)
                        print(f"  * {rel_path}")
            
            # Show quick summary
            print("\nQUICK SUMMARY:")
            for timepoint in args.timepoints:
                if timepoint in results:
                    print(f"\n{timepoint}:")
                    for method in args.methods:
                        if method in results[timepoint]:
                            acc = results[timepoint][method]['mean_accuracy']
                            ci = results[timepoint][method]['confidence_interval']
                            print(f"  {method}: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        
        else:
            print("\nNo results generated. Check your data format.")
        
    except ImportError as e:
        print(f"Error: Could not import analysis modules: {e}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
