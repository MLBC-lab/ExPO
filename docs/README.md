# ExPO Documentation

## Installation Guide

### Quick Installation

Install ExPO directly from the repository:

```bash
pip install git+https://github.com/MLBC-lab/ExPO.git
```

### Development Installation

For development or to run the latest version:

```bash
git clone https://github.com/MLBC-lab/ExPO.git
cd ExPO
pip install -e .[dev]
```

### Verify Installation

Run the setup and test script:

```bash
python setup_and_test.py
```

This will:
1. Install the package in development mode
2. Run basic functionality tests
3. Execute the demo pipeline
4. Verify all components work correctly

## Quick Start

### 1. Run the Demo

The fastest way to get started is with the built-in demo:

```bash
expo-demo
```

Or manually:
```bash
python scripts/run_full_demo_pipeline.py
```

### 2. CLI Usage

ExPO provides three main CLI commands:

#### Training
```bash
expo-train --config path/to/config.json
```

#### Evaluation
```bash  
expo-eval --config path/to/config.json --checkpoint path/to/model.pt
```

#### Demo Pipeline
```bash
expo-demo [--keep-data] [--n-profiles N] [--n-genes G]
```

### 3. Python API Usage

```python
import expo
from expo.config import ExperimentConfig
from expo.models.expo_model import ExPOModel

# Load configuration
config = ExperimentConfig.from_json("config.json")

# Create model
model = ExPOModel(n_genes=978, config=config)

# Train model
from expo.training.trainer import ExPOTrainer
trainer = ExPOTrainer(model, config)
trainer.fit(train_loader, val_loader)
```

## Configuration

ExPO uses JSON configuration files. Here's a minimal example:

```json
{
  "data": {
    "expression_table": "data/expression_table.pkl",
    "metadata_table": "data/metadata_table.pkl", 
    "compound_table": "data/compound_table.pkl",
    "split_scheme": "random",
    "n_folds": 5
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

See `demo_workspace/configs/demo_config.json` for a complete example.

## Data Format

ExPO expects three input tables:

### Expression Table
- Pickled pandas DataFrame
- Columns: `profile_id`, gene columns (e.g., `G0001`, `G0002`, ...)
- Each row is one perturbational profile

### Metadata Table  
- Pickled pandas DataFrame
- Columns: `profile_id`, `compound_id`, `cell_id`, `time`, `dose`
- Links profiles to experimental conditions

### Compound Table
- Pickled pandas DataFrame  
- Columns: `compound_id`, `smiles`
- Chemical structure information

## Advanced Usage

### Custom Data Splits
```python
from expo.data.splits import create_scaffold_split

train_idx, val_idx, test_idx = create_scaffold_split(
    compound_df, 
    train_size=0.7,
    val_size=0.15, 
    test_size=0.15
)
```

### Uncertainty Quantification
```python
from expo.calibration.conformal import MultiOutputConformalRegressor

calibrator = MultiOutputConformalRegressor(alpha=0.1)
calibrator.fit(cal_predictions, cal_targets)
intervals = calibrator.predict(test_predictions)
```

### Custom Metrics
```python
from expo.metrics.regression_metrics import compute_regression_metrics

metrics = compute_regression_metrics(
    y_true, 
    y_pred,
    per_gene_metrics=True
)
```

## Troubleshooting

### Common Issues

1. **Module not found errors**: Make sure ExPO is installed with `pip install -e .`

2. **CUDA out of memory**: Reduce batch size in config or use `"device": "cpu"`

3. **Demo pipeline fails**: Run `python setup_and_test.py` to diagnose issues

4. **Import errors**: Check that all dependencies are installed with `pip install -r requirements.txt`

### Getting Help

- Check the [GitHub Issues](https://github.com/MLBC-lab/ExPO/issues)
- Review the demo pipeline code in `scripts/run_full_demo_pipeline.py`
- Examine the synthetic data generator in `scripts/generate_test_data.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.