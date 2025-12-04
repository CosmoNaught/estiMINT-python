# estiMINT (Python)

Python port of the estiMINT R package for EIR (Entomological Inoculation Rate) estimation using machine learning.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## File Mapping (R → Python)

| R File | Python File | Description |
|--------|-------------|-------------|
| `estiMINT-package.R` | `__init__.py` | Package initialization and exports |
| `globals.R` | `globals.py` | Global variables and constants |
| `utils.R` | `utils.py` | Utility functions (metrics, QMAP, etc.) |
| `data_processing.R` | `data_processing.py` | Data loading and preprocessing |
| `models.R` | `models.py` | XGBoost model training |
| `train.R` | `train.py` | Main training pipeline with K-fold CV |
| `plotting.R` | `plotting.py` | Visualization functions |
| `storage.R` | `storage.py` | Model persistence and loading |
| `run.R` | `run.py` | Model inference |

## API Reference

### Training

```python
from estimint import train_xgb_model

model = train_xgb_model(
    in_parquet="data/input.parquet",
    out_dir="output/",
    thr_lo=0.02,           # Lower prevalence threshold
    thr_hi=0.95,           # Upper prevalence threshold
    k_strata=16,           # K-means strata for EIR
    K=10,                  # CV folds
    seed=42,
    save_pkl=True,
    save_plots=True,
    save_artifacts=True
)
```

### Inference

```python
from estimint import load_xgb_model, run_xgb_model
import pandas as pd

# Load model
model = load_xgb_model("output/models/estiMINT_model.pkl")

# Prepare input data
new_data = pd.DataFrame({
    "dn0_use": [0.5],
    "Q0": [0.3],
    "phi_bednets": [0.6],
    "seasonal": [1],
    "itn_use": [0.7],
    "irs_use": [0.2],
    "prev_y9": [0.15]  # or "prevalence"
})

# Run prediction
eir_predictions = run_xgb_model(new_data, model)
print(f"Predicted EIR: {eir_predictions[0]:.2f}")
```

### Using Global Model

```python
from estimint import load_xgb_model, run_xgb_model, set_global_model

# Set global model once
model = load_xgb_model("output/models/estiMINT_model.pkl")
set_global_model(model)

# Run predictions without passing model
predictions = run_xgb_model(new_data)  # Uses global model
```

## Utility Functions

```python
from estimint import (
    r2, rmse, mse, mae, median_ae, mae_rel, rmsle, smape,
    fit_qmap_w, predict_qmap_w, scale_pos
)

# Calculate metrics
y_true = [1, 2, 3, 4, 5]
y_pred = [1.1, 2.2, 2.9, 4.1, 4.8]

print(f"R²: {r2(y_true, y_pred):.4f}")
print(f"RMSE: {rmse(y_true, y_pred):.4f}")
print(f"MAE: {mae(y_true, y_pred):.4f}")

# Quantile mapping calibration
cal = fit_qmap_w(y_pred, y_true)
y_calibrated = predict_qmap_w(y_pred, cal)
```

## Data Processing

```python
from estimint import load_and_filter, make_value_weights, strata_and_split

# Load and filter parquet data
result = load_and_filter("data.parquet", thr_lo=0.02, thr_hi=0.95)
df = result["DT"]
df_excluded = result["DT_excluded"]

# Create inverse-frequency weights
weights = make_value_weights(df["eir"].values, digits=3)

# Stratified split
df["eir_log10"] = np.log10(df["eir"])
df = strata_and_split(df, k_strata=16, seed=42)
```

## Key Differences from R Version

1. **File format**: Models saved as `.pkl` (pickle) instead of `.rds`
2. **Data handling**: Uses pandas instead of data.table
3. **Plotting**: Uses matplotlib instead of ggplot2
4. **Global model**: Use `set_global_model()` / `get_global_model()` instead of `.GlobalEnv`

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- duckdb >= 0.8.0
- xgboost >= 1.6.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- requests >= 2.28.0 (optional, for model download)
- appdirs >= 1.4.0 (optional, for cache directory)

## License

MIT License
