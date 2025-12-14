import argparse
import numpy as np
import pandas as pd
import os
import sys

def main():
    parser = argparse.ArgumentParser(description='Predict MRSA/MSSA for new samples')
    parser.add_argument('--sample', '-s', required=True,
                       help='Path to sample file (CSV/Excel)')
    parser.add_argument('--timepoint', '-t', required=True,
                       choices=['0min', '20min', '30min', '60min'],
                       help='Timepoint for prediction')
    parser.add_argument('--method', '-m', default='LDA',
                       choices=['LDA', 'PLS_DA', 'SVM_linear', 'SVM_rbf'],
                       help='Classification method to use')
    parser.add_argument('--model-dir', default='data/models',
                       help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Check if sample file exists
    if not os.path.exists(args.sample):
        print(f"Error: Sample file not found at {args.sample}")
        sys.exit(1)
    
    try:
        from src.prediction.predictor import Predictor
        
        print("="*70)
        print("MRSA/MSSA PREDICTION")
        print("="*70)
        print(f"Sample: {args.sample}")
        print(f"Timepoint: {args.timepoint}")
        print(f"Method: {args.method}")
        print("="*70)
        
        # Initialize predictor
        predictor = Predictor(model_dir=args.model_dir)
        
        # Load sample
        if args.sample.endswith('.csv'):
            sample_df = pd.read_csv(args.sample)
        else:
            sample_df = pd.read_excel(args.sample)
        
        # Check format
        if 'target' in sample_df.columns:
            sample_data = sample_df.drop(columns=['target'])
        else:
            sample_data = sample_df
        
        # Convert to numpy array
        if sample_data.shape[0] == 1:
            X = sample_data.values.reshape(1, -1)
        else:
            X = sample_data.values
        
        print(f"Sample shape: {X.shape}")
        print(f"Expected features: 490")
        
        if X.shape[1] != 490:
            print(f"Warning: Sample has {X.shape[1]} features, expected 490")
            if X.shape[1] > 490:
                print("Using first 490 features")
                X = X[:, :490]
            else:
                print("Error: Not enough features")
                sys.exit(1)
        
        # Make prediction
        prediction = predictor.predict(
            X=X,
            timepoint=args.timepoint,
            method=args.method
        )
        
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"Predicted: {prediction['label']}")
        print(f"Probability (MRSA): {prediction['probability']:.3f}")
        print(f"Confidence: {prediction['confidence']:.1f}%")
        
        if prediction['probability'] > 0.5:
            print(f"Interpretation: MRSA (probability > 0.5)")
        else:
            print(f"Interpretation: MSSA (probability ≤ 0.5)")
        
        print("\nConfidence levels:")
        print("  ≥ 90%: High confidence")
        print("  70-89%: Moderate confidence")
        print("  < 70%: Low confidence")
        
        # Save prediction
        output_file = f"prediction_{os.path.basename(args.sample).split('.')[0]}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Sample: {args.sample}\n")
            f.write(f"Timepoint: {args.timepoint}\n")
            f.write(f"Method: {args.method}\n")
            f.write(f"Predicted: {prediction['label']}\n")
            f.write(f"Probability (MRSA): {prediction['probability']:.3f}\n")
            f.write(f"Confidence: {prediction['confidence']:.1f}%\n")
        
        print(f"\nPrediction saved to: {output_file}")
        
    except ImportError as e:
        print(f"Error: Could not import prediction modules: {e}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
