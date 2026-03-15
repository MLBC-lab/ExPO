# ExPO: Exposure–Response Neural Operator for Gene Expression Imputation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)

**ExPO** is a neural-operator–style model for learning dose–time–conditioned gene expression responses to small-molecule perturbations. This package provides a complete, reproducible, and FAIR (Findable, Accessible, Interoperable, Reusable) implementation for predicting transcriptomic responses across different exposure conditions.

## 🚀 Quick Start

### Installation

Install ExPO directly from GitHub:

```bash
pip install git+https://github.com/MLBC-lab/ExPO.git
```

For development:

```bash
git clone https://github.com/MLBC-lab/ExPO.git
cd ExPO
pip install -e .[dev]
```

### Verify Installation and Run Demo

```bash
python setup_and_test.py
```

This command will:
- ✅ Install the package and dependencies
- ✅ Run comprehensive tests
- ✅ Execute the complete demo pipeline
- ✅ Verify all functionality works correctly

### CLI Usage

ExPO provides convenient command-line tools:

```bash
# Run the demo pipeline
expo-demo

# Train a model (simplified CLI)
expo-train --config config.json

# Train a model (full functionality)
expo-train --config config.json --use-full-trainer
# OR directly:
python scripts/train_expo.py --config config.json

# Evaluate a model
expo-eval --config config.json --checkpoint model.pt
```

## 📊 Key Features

- **🎯 Dose-Time Conditional Modeling**: Predicts gene expression as a function of compound structure, dose, and time
- **🧪 Chemical Structure Encoding**: Uses ChemBERTa-style encoders for compound representation
- **📈 Comprehensive Metrics**: Regression, classification, and ranking evaluation
- **🎲 Uncertainty Quantification**: Conformal prediction and calibration methods
- **📋 FAIR Principles**: Fully reproducible with proper metadata and documentation
- **🔧 Easy to Use**: Simple CLI and Python API
- **🧪 Synthetic Data Generator**: Built-in testing with synthetic datasets

## 🏗️ Architecture

ExPO combines three key components:

1. **Chemical Encoder**: Processes molecular SMILES strings
2. **Exposure Encoder**: Handles dose and time information with Fourier features
3. **Context Integration**: Incorporates cell line and experimental metadata

The model outputs full gene expression profiles with uncertainty estimates.

## 📁 Repository Structure

```text
ExPO/
├── expo/                    # Main package
│   ├── models/              # Neural network models
│   ├── data/               # Data loading and preprocessing
│   ├── training/           # Training utilities and loops
│   ├── calibration/        # Uncertainty quantification
│   ├── metrics/            # Evaluation metrics
│   ├── analysis/           # Visualization and analysis
│   └── utils/              # Helper utilities
├── scripts/                # Executable scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
└── demo_workspace/         # Generated demo files
```

## 📖 Usage Examples

### Python API

```python
import expo
from expo.config import ExperimentConfig
from expo.models.expo_model import ExPOModel

# Load configuration
config = ExperimentConfig.from_json("config.json")

# Create and train model
model = ExPOModel(n_genes=978, config=config)
trainer = expo.training.ExPOTrainer(model, config)
trainer.fit(train_loader, val_loader)

# Make predictions
predictions = model(compound_ids, doses, times, cell_ids)
```

### Configuration

ExPO uses JSON configuration files for reproducible experiments:

```json
{
  "data": {
    "expression_table": "data/expression.pkl",
    "metadata_table": "data/metadata.pkl",
    "compound_table": "data/compounds.pkl",
    "split_scheme": "random",
    "up_threshold": 1.0,
    "down_threshold": -1.0
  },
  "training": {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 50,
    "device": "cuda"
  },
  "model": {
    "hidden_dim": 512,
    "num_layers": 3
  }
}
```

## 🔬 Data Format

ExPO expects three input tables in pandas pickle format:

### Expression Table
- **Format**: DataFrame with `profile_id` and gene columns
- **Example**: `profile_id`, `G0001`, `G0002`, ..., `G0978`

### Metadata Table  
- **Format**: DataFrame linking profiles to conditions
- **Columns**: `profile_id`, `compound_id`, `cell_id`, `time`, `dose`

### Compound Table
- **Format**: DataFrame with structure information
- **Columns**: `compound_id`, `smiles`

See the [documentation](docs/README.md) for detailed format specifications.

## 📊 Model Evaluation

ExPO provides comprehensive evaluation metrics:

- **Regression**: MAE, RMSE, R², Pearson correlation
- **Classification**: Direction-of-change accuracy, F1-score
- **Ranking**: NDCG@K, Jaccard@K for top responsive genes
- **Uncertainty**: Calibration curves, interval coverage

## 🎯 Uncertainty Quantification

Built-in support for:
- **Conformal Prediction**: Distribution-free uncertainty intervals
- **Temperature Scaling**: Calibrated classification probabilities
- **Quantile Regression**: Direct prediction interval estimation

## 🧪 Working with L1000/LINCS Data

ExPO is designed to work with LINCS L1000 Connectivity Map data:

- **Primary Sources**: NCBI GEO (GSE92742, GSE70138, GSE92743)
- **CLUE Portal**: https://clue.io/ for interactive access
- **Data Processing**: Built-in utilities for L1000 format conversion

See [Section 12 of the documentation](docs/README.md#l1000--connectivity-map-data) for detailed instructions.

## 🧪 Testing and Quality Assurance

Comprehensive testing framework:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=expo --cov-report=html

# Test specific functionality
python -m pytest tests/test_expo.py::test_synthetic_data_generation
```

## 📈 Performance Benchmarking

Compare ExPO against baseline methods:

```python
from expo.baselines import ConstantBaseline, DoseScaledBaseline
from expo.metrics import compare_models

baselines = [
    ConstantBaseline(),
    DoseScaledBaseline(),
    ExPOModel(config=config)
]

comparison = compare_models(baselines, test_data)
```

## 🔧 Development and Contributing

### Development Setup

```bash
git clone https://github.com/MLBC-lab/ExPO.git
cd ExPO
pip install -e .[dev]
pre-commit install  # Install git hooks
```

### Code Quality

- **Formatting**: Black (automatically applied)
- **Linting**: Flake8 for style checking  
- **Type Checking**: MyPy for static analysis
- **Testing**: Pytest with coverage reporting

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with appropriate tests
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## 📚 Documentation

- **[Installation Guide](docs/README.md)**: Detailed setup instructions
- **[API Reference](docs/API.md)**: Complete function documentation
- **[Examples](scripts/)**: Real-world usage examples
- **[Configuration](demo_workspace/configs/)**: Sample configuration files

## 🏆 Citation

If you use ExPO in your research, please cite:

```bibtex
@article{expo2024,
  title={ExPO: Exposure-Response Neural Operator for Gene Expression Imputation},
  author={MLBC Lab},
  journal={Under Review},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

ExPO builds upon:
- **Neural Operators**: For structured prediction over continuous domains
- **ChemBERTa**: Chemical language models for molecular representation
- **LINCS L1000**: Large-scale transcriptomic perturbation data
- **Conformal Prediction**: Distribution-free uncertainty quantification

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/MLBC-lab/ExPO/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MLBC-lab/ExPO/discussions)
- **Email**: info@mlbc-lab.org

## 📊 Project Status

| Component | Status |
|-----------|--------|
| ✅ Core Model | Complete |
| ✅ Training Pipeline | Complete |
| ✅ Evaluation Metrics | Complete |
| ✅ Uncertainty Quantification | Complete |  
| ✅ Documentation | Complete |
| ✅ Testing Suite | Complete |
| ✅ CLI Interface | Complete |
| ✅ Demo Pipeline | Complete |

**ExPO is ready for research and production use!** 🎉

