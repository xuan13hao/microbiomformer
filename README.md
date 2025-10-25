# Microbiomformer

A transformer-based deep learning model for analyzing human gut microbiome data to predict immunotherapy response in lung cancer patients.

## Overview

Microbiomformer is a state-of-the-art machine learning pipeline that combines transformer architecture with multilayer perceptron (MLP) models to analyze complex microbiome datasets. The project focuses on predicting immunotherapy response in non-small cell lung cancer (NSCLC) patients using their gut microbiome composition.

## Key Features

- **Transformer-based Architecture**: Utilizes advanced transformer models for high-performance microbiome data analysis
- **Human Gut Microbiome Focus**: Specially designed for human gut microbiome datasets with domain-specific preprocessing
- **Immunotherapy Response Classification**: Integrates with MLP models to classify responses to lung cancer immunotherapy
- **Scalable and Customizable**: Easily adaptable to different datasets and research needs
- **Comprehensive Analysis Pipeline**: From data preprocessing to model training and evaluation

## Project Structure

```
microbiomformer/
├── data/                          # Data files
│   ├── metadata.csv              # Patient metadata with clinical information
│   ├── metadata_response.csv     # Binary response labels (0/1)
│   ├── merged_table.csv          # Combined microbiome and response data
│   ├── genus_rotated_f_filtered.csv  # Filtered genus-level microbiome data
│   ├── otu_table_normalized.csv  # Normalized OTU count data
│   ├── normalized_otu_counts.csv # Normalized microbiome counts
│   └── ...                       # Other processed data files
├── notebooks/                     # Jupyter notebooks
│   ├── pretrain.ipynb           # Transformer pretraining notebook
│   ├── train_mlp.ipynb          # MLP training with pretrained features
│   ├── Analysis.ipynb           # Comprehensive data analysis
│   └── clean_data.ipynb         # Data preprocessing and cleaning
├── models/                       # Trained model files
│   └── pretrained_model.pth     # Pretrained transformer model
├── scripts/                      # Python scripts
│   └── TestUsingGivenProfile.py # Model testing script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/microbiomformer.git
   cd microbiomformer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preprocessing

Start with the data cleaning notebook to prepare your microbiome data:

```bash
jupyter notebook notebooks/clean_data.ipynb
```

### 2. Transformer Pretraining

Train the transformer model on microbiome data:

```bash
jupyter notebook notebooks/pretrain.ipynb
```

This notebook:
- Loads and preprocesses microbiome data
- Implements masked language modeling for microbiome sequences
- Trains a transformer encoder model
- Saves the pretrained model weights

### 3. MLP Training

Train the classification model using pretrained features:

```bash
jupyter notebook notebooks/train_mlp.ipynb
```

This notebook:
- Loads the pretrained transformer model
- Extracts features for classification
- Trains an MLP classifier for immunotherapy response prediction
- Evaluates model performance

### 4. Comprehensive Analysis

Run the analysis notebook for detailed insights:

```bash
jupyter notebook notebooks/Analysis.ipynb
```

### 5. Model Testing

Use the testing script for inference on new data:

```bash
python scripts/TestUsingGivenProfile.py <profile> <report_name> <sp_code> <feature1> <feature2> ...
```

## Data Format

### Input Data
- **Microbiome Data**: Genus-level abundance data (CSV format)
- **Metadata**: Patient clinical information including age, BMI, sex, response status
- **Response Labels**: Binary classification (0: non-responder, 1: responder)

### Model Architecture

1. **Transformer Encoder**: 
   - Input dimension: Variable (based on number of genera)
   - Hidden dimension: 64
   - Number of heads: 8
   - Number of layers: 6

2. **MLP Classifier**:
   - Input: Extracted transformer features
   - Hidden layers: 128 neurons
   - Output: Binary classification

## Performance

The model achieves competitive performance on immunotherapy response prediction:
- **Accuracy**: ~70-80% (varies by dataset)
- **F1 Score**: ~0.6-0.8
- **Cross-validation**: 5-fold stratified validation

## Key Notebooks Explained

### `pretrain.ipynb`
- **Purpose**: Pretrains a transformer model on microbiome data using masked language modeling
- **Key Features**: 
  - Custom OTU dataset class with masking
  - Transformer encoder architecture
  - Training loop with loss visualization
  - Model checkpointing

### `train_mlp.ipynb`
- **Purpose**: Trains MLP classifier using pretrained transformer features
- **Key Features**:
  - Feature extraction from pretrained model
  - Enhanced MLP with transformer integration
  - Performance evaluation and metrics
  - Comparison with baseline models

### `Analysis.ipynb`
- **Purpose**: Comprehensive data analysis and model evaluation
- **Key Features**:
  - Data exploration and visualization
  - Statistical analysis of microbiome differences
  - Model performance comparison
  - Feature importance analysis

### `clean_data.ipynb`
- **Purpose**: Data preprocessing and cleaning pipeline
- **Key Features**:
  - Data quality assessment
  - Normalization and filtering
  - Feature engineering
  - Data validation

## Dependencies

- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib/seaborn**: Visualization
- **scipy**: Scientific computing

## Contributing

We welcome contributions to Microbiomformer! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

<!-- ## Citation

If you use Microbiomformer in your research, please cite:

```bibtex
@software{microbiomformer2024,
  title={Microbiomformer: A Transformer-based Model for Microbiome Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/microbiomformer}
}
``` -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to:
- Dr. Zhong for model support and guidance
- Tianxiao Zhang for research collaboration
- The research team for invaluable support and contributions
- The open-source community for the foundational tools and libraries

## Contact

For questions, issues, or collaborations, please:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

## Future Work

- [ ] Support for additional microbiome data formats
- [ ] Integration with other omics data types
- [ ] Web interface for model deployment
- [ ] Extended validation on multiple cancer types
- [ ] Real-time prediction capabilities