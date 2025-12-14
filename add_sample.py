import argparse
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Add new samples to database')
    parser.add_argument('--sample', '-s', required=True,
                       help='Path to sample file (CSV/Excel)')
    parser.add_argument('--label', '-l', required=True,
                       help='Sample label (e.g., "MRSA_0min", "MSSA_30min")')
    parser.add_argument('--user', '-u', default='anonymous',
                       help='User who submitted the sample')
    parser.add_argument('--comments', '-c', default='',
                       help='Additional comments')
    parser.add_argument('--database', '-d', default='data/samples_database.parquet',
                       help='Path to database file')
    
    args = parser.parse_args()
    
    # Check if sample file exists
    if not os.path.exists(args.sample):
        print(f"Error: Sample file not found at {args.sample}")
        sys.exit(1)
    
    try:
        from src.database.manager import SampleDatabase
        
        print("="*70)
        print("ADD SAMPLE TO DATABASE")
        print("="*70)
        print(f"Sample: {args.sample}")
        print(f"Label: {args.label}")
        print(f"User: {args.user}")
        print("="*70)
        
        # Load sample
        if args.sample.endswith('.csv'):
            sample_df = pd.read_csv(args.sample)
        else:
            sample_df = pd.read_excel(args.sample, engine='odf')
        
        # Extract spectra (first 490 columns)
        spectra_columns = []
        for col in sample_df.columns:
            try:
                float(col)
                spectra_columns.append(col)
            except:
                if len(spectra_columns) < 490:
                    spectra_columns.append(col)
        
        if len(spectra_columns) < 490:
            spectra_columns = list(sample_df.columns[:490])
        
        spectra = sample_df[spectra_columns].values
        
        print(f"Sample shape: {spectra.shape}")
        print(f"Features: {len(spectra_columns)}")
        
        # Initialize database
        db = SampleDatabase(db_path=args.database)
        
        # Add sample
        sample_id = db.add_sample(
            spectra=spectra,
            label=args.label,
            submitted_by=args.user,
            comments=args.comments
        )
        
        print("\n" + "="*70)
        print("SAMPLE ADDED SUCCESSFULLY")
        print("="*70)
        print(f"Sample ID: {sample_id}")
        print(f"Database: {args.database}")
        print(f"Total samples: {db.count_samples()}")
        print(f"Pending review: {db.count_pending()}")
        
        # Optional: Make a prediction on the new sample
        response = input("\nWould you like to predict this sample? (y/n): ")
        if response.lower() == 'y':
            from src.prediction.predictor import Predictor
            
            predictor = Predictor(model_dir='data/models')
            
            # Extract timepoint from label
            import re
            match = re.search(r'(\d+)min', args.label)
            if match:
                timepoint = f"{match.group(1)}min"
                method = 'LDA'  # Default method
                
                prediction = predictor.predict(
                    X=spectra,
                    timepoint=timepoint,
                    method=method
                )
                
                print(f"\nPrediction for {timepoint} with {method}:")
                print(f"  Label: {prediction['label']}")
                print(f"  Probability: {prediction['probability']:.3f}")
                print(f"  Confidence: {prediction['confidence']:.1f}%")
                
                # Update sample with prediction
                db.update_sample(
                    sample_id=sample_id,
                    predicted_label=prediction['label'],
                    probability=prediction['probability'],
                    confidence=prediction['confidence']
                )
        
        print(f"\nSample added with ID: {sample_id}")
        
    except ImportError as e:
        print(f"Error: Could not import database modules: {e}")
        print("\nPlease install required packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error adding sample: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
