#!/usr/bin/env python3
"""
Run the modular analysis with clean output
"""
import sys
import os

# Add src to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from Analysis.modular_pipeline import AnalysisPipeline

def main():
    # Create pipeline with verbose output
    pipeline = AnalysisPipeline(verbose=True)
    
    # Run analysis
    data_path = 'Data/processed_spectra_normalized_2nd_derivative_balenced.ods'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure your data file is in the Data directory")
        return
    
    print("=" * 80)
    print("RUNNING MODULAR ANALYSIS WITH CLEAN OUTPUT")
    print("Only shows progress bars and final results")
    print("=" * 80)
    
    results = pipeline.run_analysis(data_path)
    
    if results:
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nResults saved in 'modular_results' directory:")
        print("  • modular_results/summary_*.csv - Summary statistics")
        print("  • modular_results/*_accuracies_*.csv - Detailed accuracy values")
        print("  • modular_results/modular_analysis_results.png - Plot")
        
        # Print quick summary
        print("\nQUICK SUMMARY:")
        for timepoint in ['0min', '20min', '30min', '60min']:
            if timepoint in results:
                print(f"\n{timepoint}:")
                for method in ['LDA', 'PLS_DA', 'SVM_linear', 'SVM_rbf']:
                    if method in results[timepoint]:
                        acc = results[timepoint][method]['mean_accuracy']
                        ci = results[timepoint][method]['confidence_interval']
                        print(f"  {method}: {acc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    else:
        print("\n✗ No results generated")

if __name__ == "__main__":
    main()
