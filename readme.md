# MRSA Predictor

A machine learning tool for predicting MRSA/MSSA from FTIR spectra with biological feature selection.

## Features

- **LASSO Stability Analysis**: Selects robust features using bootstrap stability
- **Multiple Classifiers**: LDA, PLS-DA, SVM (linear), SVM (rbf)
- **Strain-Preserving Splits**: Maintains strain diversity in train/test splits
- **Easy to Use**: Simple CLI interface for analysis and prediction

## Quick Start

### Installation

git clone https://github.com/Jiang-Lab/FTIR-MRSA/
cd mrsa-predictor
pip install -r requirements.txt

### Examples on how to use:
use --quick for faster but less accurate results (analysis can take a long time)

##### Analyze your existing data
python analyze_data.py --data your_data.ods

##### Or use the example data
python analyze_data.py

##### Predict sample
python predict_sample.py --sample new_spectra.csv --timepoint 0min

##### Add sample to database
python add_sample.py --sample new_spectra.csv --label "MRSA_0min" --user "researcher"

*Note: gets added to a temporary database untill confirmed*

## Data Formating
use csv file
