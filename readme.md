# FTIR-MRSA: Rapid Discrimination of MRSA and MSSA

A machine learning pipeline for rapid discrimination of methicillin-resistant *Staphylococcus aureus* (MRSA) from methicillin-sensitive *S. aureus* (MSSA) using ATR-FTIR spectroscopy. This tool achieves **91% accuracy within 20 minutes** of antibiotic exposure.


## Overview
This repository implements the methods described in our paper: ** "Early cell-wall stress signatures enable 20-minute ATR-FTIR discrimination of MRSA and MSSA"**. Traditional susceptibility testing requires 48-72 hours, but our approach leverages early biochemical signatures detected by Fourier-transform infrared (FTIR) spectroscopy combined with interpretable machine learning to deliver rapid results.

## Key Features

_**Rapid Detection**: Accurate MRSA/MSSA discrimination at 20-30 minutes post-antibiotic exposure
_**High Accuracy**: 91% accuracy at 20 min, 92% at 30 min using LASSO-selected features
_**Interpretable Models**: Sparse linear classifiers with biologically meaningful features
_**Mechanistic Insight**: Features map to peptidoglycan, carbohydrate, and lipid regions reflecting cell-wall stress
_**Multiple Classifiers**: Support for LDA, PLS-DA, Linear SVM, and RBF SVM

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Usage Examples](#usage-examples)
- [Pipeline Overview](#pipeline-overview)
- [Results & Validation](#results--validation)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

##  Installation

### Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/Jiang-Lab/FTIR-MRSA.git
cd FTIR-MRSA

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core packages include:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Scientific computing and signal processing
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization

## Quick Start

### Analyze Existing Data

```bash
# Use the included example data
python analyze_data.py

# Or analyze your own data
python analyze_data.py --data your_data.ods

# Quick analysis (faster, less rigorous validation)
python analyze_data.py --quick
```

### Predict New Samples

```bash
# Predict MRSA/MSSA classification for new spectra
python predict_sample.py --sample new_spectra.csv --timepoint 20min

# Supported timepoints: 0min, 20min, 30min, 60min
```

### Add Samples to Database

```bash
# Add new validated samples (feature in development)
python add_sample.py --sample new_spectra.csv --label "MRSA_20min" --user "researcher_id"
```

## Data Format

### Input Spectra Format

Data should be provided as CSV or ODS files with the following structure:

```csv
wavenumber,absorbance
1800,0.0234
1799,0.0245
...
800,0.0156
```

**Requirements:**
- Wavenumber range: 800-1800 cm⁻¹
- Second-derivative preprocessed spectra (or raw absorbance for preprocessing)
- One spectrum per file or multiple columns for batch processing

##  Usage Examples

### 1. Full Analysis Pipeline

```python
from src.preprocessing import load_and_preprocess
from src.feature_selection import lasso_stability_selection
from src.classifiers import train_and_evaluate

# Load data
X, y, metadata = load_and_preprocess('data/spectra.ods', timepoint='20min')

# Feature selection
selected_features = lasso_stability_selection(X, y, stability_threshold=0.2)

# Train and evaluate
results = train_and_evaluate(X[:, selected_features], y, 
                             method='linear_svm',
                             cv_strategy='strain_aware')

print(f"Accuracy: {results['accuracy']:.3f}")
print(f"Selected features: {len(selected_features)}")
```

### 2. Visualize Spectral Features

```python
from src.visualization import plot_spectra, plot_lasso_stability

# Plot mean spectra by class
plot_spectra('data/spectra.ods', timepoint='20min', 
             output='results/figures/spectra_20min.png')

# Visualize LASSO feature selection
plot_lasso_stability('data/spectra.ods', 
                     output='results/figures/lasso_stability.png')
```

### 3. Time-Course Analysis

```bash
# Analyze all timepoints
python analyze.py --data data/spectra.ods --timepoints all --output results/
```

## Pipeline Overview

### 1. Preprocessing

- **Normalization**: Min-max normalization (1000-1800 cm⁻¹)
- **Smoothing**: 11-point Savitzky-Golay filter
- **Derivative**: Second derivative for peak resolution
- **Standardization**: Zero mean, unit variance per feature

### 2. Feature Selection

**LASSO with Bootstrap Stability Analysis:**

- 100 bootstrap iterations
- λ = 0.05 regularization parameter
- Features selected if stability ≥ 20%
- Focuses on biologically relevant regions:
  - Peptidoglycan precursors (950-1100 cm⁻¹)
  - Amide I (1560-1685 cm⁻¹)
  - Lipid membranes (1725-1745 cm⁻¹)
  - Tyrosine bands (1500-1530 cm⁻¹)

### 3. Classification

**Supported Algorithms:**
- Linear Discriminant Analysis (LDA)
- Partial Least Squares-DA (PLS-DA)
- Linear Support Vector Machine (SVM)
- RBF SVM

**Validation Strategy:**
- Strain-aware cross-validation
- Minimum 1 sample per strain in each split
- Class balance maintained
- 12-15 samples per training set

## Results & Validation

### Performance Metrics

| Timepoint | Feature Set | Classifier | Accuracy | Sensitivity | Specificity |
|-----------|-------------|------------|----------|-------------|-------------|
| 20 min    | LASSO       | Linear SVM | 0.91     | 95.83%      | 73.86%      |
| 30 min    | LASSO       | Linear SVM | 0.92     | -           | -           |
| 60 min    | LASSO       | Linear SVM | 0.82     | -           | -           |

### Validated Strains

- **MSSA**: ATCC 6538, RN4220
- **MRSA**: ATCC 43300, USA300-JE2

### Discriminative Biomarkers

**20-30 minutes (Early):**
- 1516 cm⁻¹ - Peptidoglycan
- 1025 cm⁻¹ - Carbohydrate precursors
- Cell-wall stress signatures

**60 minutes (Late):**
- 1735 cm⁻¹ - Lipid ester C=O
- Amide I/II bands
- Membrane perturbation

### Orthogonal Validation

- **TEM**: Confirmed cell-wall thickening in MSSA at 20 min
- **AFM**: Demonstrated decreased cell height in MSSA under ampicillin
- **Growth curves**: Early stagnation/decline in MSSA OD₆₀₀

## Citation

If you use this code or method in your research, please cite:

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Clinical isolate validation
- [ ] Additional antibiotic classes
- [ ] Automated sample handling integration
- [ ] Real-time spectral acquisition interface
- [ ] Multi-site validation protocols
- [ ] Docker containerization

## ⚠️ Limitations & Future Work

### Current Limitations

- Limited to 4 laboratory strains
- Single β-lactam (ampicillin) tested
- ~15 minute drying time required
- Validated only up to 60 minutes

### Planned Improvements

- **Clinical Translation**: Testing on diverse clinical isolates
- **Additional Antibiotics**: Extend to glycopeptides, lipopeptides
- **Automated Workflows**: Integration with sample handling systems
- **Multi-site Validation**: Cross-laboratory reproducibility studies
- **Real-time Processing**: Online spectral analysis capabilities

## Contact

Yi Jiang (yjiang12@gsu.edu)   
Wilbur Hudson (whudson7@gsu.edu) 

**Institution:**  
Georgia State University  
Department of Mathematics and Statistics  
Atlanta, GA 30303, USA

## Acknowledgments

- Frady Whipple Endowed Professorship (to YJ)
- Molecular Basis of Disease (MBD) graduate fellowship (to DS)
- RISE Challenge Grant, Georgia State University
- Emory University Robert P. Apkarian Integrated Electron Microscopy Core Facility
- Georgia Clinical & Translational Science Alliance, NIH

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Resources

- [Paper Repository](https://github.com/Jiang-Lab/FTIR-MRSA)
- [Lab Website](https://math.gsu.edu/yjiang12)


**Note:** This is a research tool validated on laboratory strains. Clinical deployment requires additional validation on diverse clinical isolates and regulatory approval.
