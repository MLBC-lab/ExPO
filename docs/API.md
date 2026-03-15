# API Reference

## Core Components

### ExperimentConfig

Configuration management for ExPO experiments.

```python
from expo.config import ExperimentConfig

# Load from JSON
config = ExperimentConfig.from_json("config.json")

# Create programmatically
config = ExperimentConfig(
    data={
        "expression_table": "data/expr.pkl",
        "metadata_table": "data/meta.pkl", 
        "compound_table": "data/compounds.pkl"
    },
    training={
        "learning_rate": 1e-3,
        "batch_size": 32
    }
)
```

### ExPOModel

Main neural network model.

```python
from expo.models.expo_model import ExPOModel

model = ExPOModel(
    n_genes=978,
    config=config
)

# Forward pass
predictions = model(compound_ids, doses, times, cell_ids)
```

## Data Handling

### ExPODataset

Dataset class for loading and preprocessing data.

```python
from expo.data.dataset import ExPODataset

dataset = ExPODataset(
    expression_df=expr_df,
    metadata_df=meta_df,
    compound_df=comp_df,
    config=config
)
```

### Data Preprocessing

```python
from expo.data.preprocessing import preprocess_expression_data

processed_expr = preprocess_expression_data(
    expr_df, 
    normalization="z_score",
    clip_values=(-5, 5)
)
```

## Training

### ExPOTrainer

Training loop implementation.

```python
from expo.training.trainer import ExPOTrainer

# For full training, use the complete script
trainer = ExPOTrainer(model, config, train_loader, val_loader)
trainer.fit(train_loader, val_loader)

# Or use the training script directly
import subprocess
result = subprocess.run([
    "python", "scripts/train_expo.py", 
    "--config", "config.json"
])
```

### Custom Training Loop

```python
import torch
from expo.losses.regression import HuberLoss

criterion = HuberLoss(delta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in train_loader:
    predictions = model(**batch)
    loss = criterion(predictions, batch['targets'])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Metrics and Evaluation

### Regression Metrics

```python
from expo.metrics.regression_metrics import compute_regression_metrics

metrics = compute_regression_metrics(
    y_true, 
    y_pred,
    per_gene_metrics=True,
    top_k_genes=[10, 50, 100]
)

# Available metrics:
# - mae, rmse, r2, pearson_r
# - per_gene_mae, per_gene_r2
# - top_k_mae, top_k_r2
```

### Classification Metrics

```python
from expo.metrics.classification_metrics import compute_classification_metrics

# For direction-of-change classification
class_metrics = compute_classification_metrics(
    y_true_classes,
    y_pred_classes,
    average="macro"
)
```

### Ranking Metrics

```python
from expo.metrics.ranking_metrics import compute_ranking_metrics

ranking_metrics = compute_ranking_metrics(
    y_true,
    y_pred, 
    top_k=[10, 50, 100]
)

# Includes: NDCG@K, Jaccard@K, RBO
```

## Uncertainty and Calibration

### Conformal Prediction

```python
from expo.calibration.conformal import MultiOutputConformalRegressor

# Fit on calibration set
calibrator = MultiOutputConformalRegressor(alpha=0.1)
calibrator.fit(cal_predictions, cal_targets)

# Get prediction intervals
intervals = calibrator.predict(test_predictions)
```

### Temperature Scaling

```python
from expo.calibration.temperature_scaling import TemperatureScaler

scaler = TemperatureScaler()
scaler.fit(logits, targets)
calibrated_probs = scaler(new_logits)
```

### Reliability Curves

```python
from expo.calibration.reliability import regression_reliability_curve

bin_centers, bin_accuracies = regression_reliability_curve(
    y_true,
    y_pred,
    n_bins=10
)
```

## Visualization

### Training Curves

```python
from expo.analysis.plotting import plot_training_curves

plot_training_curves(
    metrics_jsonl_path="runs/experiment/metrics.jsonl",
    save_dir="figures/"
)
```

### Calibration Plots

```python
from expo.analysis.calibration_plots import plot_reliability_curve

plot_reliability_curve(
    y_true,
    y_pred,
    save_path="calibration.png"
)
```

### Embedding Analysis

```python
from expo.analysis.embedding_analysis import analyze_compound_embeddings

embeddings_analysis = analyze_compound_embeddings(
    model,
    compound_df,
    save_dir="analysis/"
)
```

## Utilities

### Checkpointing

```python
from expo.utils.checkpoints import save_checkpoint, load_checkpoint

# Save
save_checkpoint(
    model,
    optimizer, 
    epoch,
    metrics,
    "checkpoint.pt"
)

# Load
checkpoint = load_checkpoint("checkpoint.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

### Reproducible Seeds

```python
from expo.training.seed import set_seed

set_seed(42)  # Sets seeds for numpy, torch, random
```

### Profiling

```python
from expo.utils.profiling import profile_model

profile_results = profile_model(
    model,
    sample_input,
    num_runs=100
)
```

## Advanced Features

### Custom Loss Functions

```python
from expo.losses.composite import CompositeLoss

loss_fn = CompositeLoss([
    ("mse", torch.nn.MSELoss(), 1.0),
    ("ranking", ListNetLoss(), 0.1), 
    ("regularization", L2Regularization(), 0.01)
])
```

### Multi-Output Quantile Regression

```python
from expo.models.quantile import QuantileExPOModel

quantile_model = QuantileExPOModel(
    n_genes=978,
    quantiles=[0.1, 0.5, 0.9],
    config=config
)
```

### Custom Encoders

```python
from expo.models.chem_encoder import ChemicalEncoder

encoder = ChemicalEncoder(
    model_name="chemberta-small",
    hidden_dim=512,
    freeze_weights=True
)