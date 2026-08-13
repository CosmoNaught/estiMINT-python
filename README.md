# estiMINT

estiMINT estimates malaria transmission intensity from malaria prevalence or
human biting rate (HBR), accounting for intervention coverage. Its pretrained
models are conditional rational-quadratic spline (RQS) flows: they provide a
median prediction, arbitrary quantiles, and prediction intervals with optional
conformal calibration.

The package also provides:

- EIR estimation from year-9 prevalence or HBR
- EIR-to-HBR conversion and projected EIR changes from mosquito-density changes
- Bednet resistance and coverage conversion to the `dn0` transmission covariate
- A `run_scenarios` pipeline that combines estiMINT estimates with the stateMINT
  prevalence and case-burden emulator

## Installation

estiMINT requires Python 3.12 or newer.

```bash
pip install estimint
```

Optional extras are available for specific workflows:

```bash
pip install "estimint[train]"       # training and data preparation
pip install "estimint[gpu]"         # CUDA-enabled JAX
pip install "estimint[viz]"         # plotting
pip install "estimint[scenarios]"   # stateMINT scenario emulator
pip install "estimint[all]"         # train, viz, download, scenarios. gpu
```

For local development with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras --dev
```

The RQS weights are downloaded from Hugging Face when first requested and are
cached by `huggingface_hub`. The high-level scenario pipeline additionally
loads stateMINT only when `run_scenarios` is called.

## Pretrained RQS models

The default artifacts are hosted in [`dide-ic/estiMINT`](https://huggingface.co/dide-ic/estiMINT).

| Artifact | Input | Output |
| --- | --- | --- |
| `prev_y9-eir` | year-9 prevalence (`prev_y9`) | EIR (`eir`) |
| `hbr_y9-eir` | year-9 HBR (`hbr_y9`) | EIR (`eir`) |
| `eir-hbr_y9` | EIR (`eir`) | year-9 HBR (`hbr_y9`) |

All artifacts use the supplied transmission measure plus these intervention
covariates: `dn0_use`, `Q0`, `phi_bednets`, `seasonal`, `itn_use`, and
`irs_use`. Pass raw values; feature scaling and any required log transforms
are applied by the artifact.

### Direct inference

```python
from estimint.v2.models.rqs import ConditionalRQS

artifact = ConditionalRQS.from_pretrained(
    "dide-ic/estiMINT",
    predictor="prev_y9",
    target="eir",
)

inputs = {
    "prev_y9": 0.30,
    "dn0_use": 0.33,
    "Q0": 0.87,
    "phi_bednets": 0.82,
    "seasonal": 0.0,
    "itn_use": 0.60,
    "irs_use": 0.0,
}

median_eir = artifact.predict(inputs)[0]
upper_quantile_eir = artifact.quantile(inputs, 0.90)[0]
lower_eir, upper_eir = artifact.interval(inputs, alpha=0.10)  # arrays, one entry per row
```

`predict()` returns the median. `quantile()` evaluates a requested probability
level, and `interval(alpha=0.10)` returns the lower and upper bounds of a 90%
prediction interval. All three return one value per input row.
Inputs may be a single feature dictionary, a list of dictionaries, or a NumPy
array whose columns are already in `artifact.feature_names` order. Dictionary
rows must contain exactly the artifact's expected features; this prevents
accidental feature-order or feature-name mismatches.

`interval()` widens the raw quantile band by the conformal offset stored in
`artifact.conformal`, which training calibrates for `alpha=0.10` only. If an
artifact carries no offset for the requested `alpha` the offset is zero and the
returned band is the uncorrected model quantile band. Inspect
`artifact.conformal` to see which levels a given artifact has calibrated; the
artifacts currently published on the Hub carry none.

To load an exported local artifact instead, pass its directory to
`from_pretrained`:

```python
artifact = ConditionalRQS.from_pretrained(
    "artifacts/prev_y9-eir",
    predictor="prev_y9",
    target="eir",
)
```

## Bednet covariates

`calculate_dn0` converts an insecticide-resistance level and a bednet-usage
mix into the `dn0` killing parameter and total ITN use.

```python
from estimint import calculate_dn0, net_types

net_types()
result = calculate_dn0(0.5, py_only=0.4, py_pbo=0.3, py_pyrrole=0.2, py_ppf=0.1)

print(result.dn0, result.itn_use)
```

The short names `py_only`, `py_pbo`, `py_pyrrole`, and `py_ppf` are accepted,
as are their canonical `pyrethroid_*` names.

## Run scenarios

`run_scenarios` estimates EIR and then runs the stateMINT emulator. A scenario
can start from prevalence, HBR, or a supplied EIR. A mosquito-density change is
applied only for prevalence inputs: estiMINT estimates baseline HBR, scales it
by `1 + mosquito_delta`, converts it back to EIR, and preserves the baseline
EIR estimate through that relative change.

```python
from estimint import EirTarget, Scenario, run_scenarios

scenarios = [
    Scenario(
        name="PBO campaign with higher mosquito density",
        eir_target=EirTarget(0.30, "prevalence"),
        res_use=0.55,
        py_pbo=0.85,
        Q0=0.90,
        phi=0.85,
        seasonal=1.0,
        irs=0.40,
        mosquito_delta=0.60,
        net_type_future="pyrethroid_pbo",
        itn_future=0.85,
        irs_future=0.40,
    ),
    Scenario(
        name="HBR input",
        eir_target=EirTarget(250000.0, "hbr"),
        res_use=0.45,
        py_only=0.30,
        py_ppf=0.20,
        Q0=0.80,
        phi=0.82,
        seasonal=0.0,
        irs=0.0,
    ),
    Scenario(
        name="Supplied EIR",
        eir_target=EirTarget(20.0, "eir"),
        res_use=0.0,
        Q0=0.88,
        phi=0.78,
        seasonal=1.0,
        irs=0.60,
    ),
]

results = run_scenarios(scenarios)
print(results[["name", "eir_baseline", "eir_final", "prev_y9", "cases_endline"]])
```

Each `Scenario` requires `name`, `res_use`, `Q0`, `phi`, `seasonal`, `irs`, and
an `EirTarget`. Current bednet coverage is represented by a mix of `py_only`,
`py_pbo`, `py_pyrrole`, and `py_ppf`. The optional future leg is separate:
set both `net_type_future` and `itn_future` to specify future nets. It does not
inherit the current net mix. `irs_future`, `routine`, and `lsm` each default to
zero. PPF coverage additionally contributes to the emulator's LSM covariate.

The returned DataFrame has one row per scenario and includes:

- scenario and intervention covariates, including `dn0_use` and `dn0_future`
- `eir_baseline` and `eir_final`
- `hbr_baseline` and `hbr_new` for prevalence scenarios with a mosquito change
- `prev_y9`, `prev_endline`, and `cases_endline`
- 157-step `prevalence` and non-negative `cases` NumPy arrays

Call `preload_models()` before repeated scenario runs to download and cache the
estiMINT and stateMINT models explicitly.

## Training workflow

> These commands require the training extra or a development installation:
> `pip install "estimint[train]"` or `uv sync --all-extras --dev`.

RQS training and artifact export code lives in `estimint.v2`. The standard
workflow is:

1. Prepare a simulation parquet dataset and train one of the three mappings.
2. Evaluate the generated test metrics and prediction-interval coverage.
3. Export the checkpoint and fitted preprocessing metadata as an inference
   artifact.
4. Upload the artifact to Hugging Face.

### 1. Train a model

The default configuration in `estimint.v2.conf.train_config` trains the
`prev_y9 -> eir` model from `datasets/estimint_simulations_y9.parquet`.
It expects simulation identifiers (`parameter_index`, `simulation_index`), the
model's predictor and target columns, and the six intervention covariates
listed in [Pretrained RQS models](#pretrained-rqs-models).

```bash
uv run python -m estimint.v2.train_base \
  predictor=prev_y9 \
  target=eir \
  output_dir=train_outputs/prev_y9-eir \
  use_wandb=false
```

Train the remaining deployed mappings by changing `predictor` and `target`:

```bash
uv run python -m estimint.v2.train_base predictor=hbr_y9 target=eir
uv run python -m estimint.v2.train_base predictor=eir target=hbr_y9
```

Useful overrides include `data_file`, `split_file`, `use_existing_split`,
`stratify`, `num_epochs`, `batch_size`, `lr`, and the RQS architecture settings
`width`, `depth`, `n_bins`, `rqs_bounds`, `mlp_residual`, and `dropout_rate`.
Hydra prints the resolved configuration before training begins.

Training creates or reuses the configured split CSV, writes fitted scalers to
the output directory, calculates the conformal offset, and saves the Orbax
checkpoint. With the default time-based run ID, the important outputs are:

```text
train_outputs/prev_y9-eir/
|-- features_scaler.pkl
|-- target_scaler.pkl
|-- conformal-<run-id>.json
`-- ckpts-<run-id>/
    `-- RQS/
```

Use the same `<run-id>` when exporting. The architecture arguments given to
export must exactly match the checkpoint's training configuration.

### 2. Export an inference artifact

Set the predictor, target, and training run ID. The default architecture values
already match the default training configuration; pass explicit architecture
overrides when the model was trained with non-default values.

```bash
RUN_ID=2026-08-13T12:00:00

uv run python -m estimint.v2.model_export \
  predictor=prev_y9 \
  target=eir \
  timestamp="$RUN_ID" \
  output_dir=train_outputs/prev_y9-eir \
  artifact_dir=artifacts/prev_y9-eir
```

The exporter restores the selected checkpoint, embeds the feature and target
scalers plus conformal offsets in `config.json`, and writes a portable artifact:

```text
artifacts/prev_y9-eir/
|-- config.json
`-- checkpoint/
    `-- RQS/
```

Load this directory locally with `ConditionalRQS.from_pretrained`, as shown in
[Direct inference](#direct-inference). The artifact contains all inference
preprocessing metadata, so the training dataset and pickle scaler files are not
needed at prediction time.

### 3. Upload to Hugging Face

Authenticate with an account that can write to the target model repository:

```bash
hf auth login
```

Upload the artifact under a subdirectory named exactly
`<predictor>-<target>`. This layout is required by
`ConditionalRQS.from_pretrained` when it downloads an artifact from the Hub.

```bash
hf upload dide-ic/estiMINT \
  artifacts/prev_y9-eir \
  prev_y9-eir/ \
  --commit-message "Add prev_y9 to EIR RQS artifact"
```

Repeat this for `hbr_y9-eir` and `eir-hbr_y9` when publishing the complete
model set. To pin a set of Hub artifacts for reproducible inference, create a
repository tag and pass it as the `revision` argument to `from_pretrained`:

```bash
hf repos tag create dide-ic/estiMINT v1.0.0 \
  --revision main \
  --message "Release RQS model set v1.0.0"
```

```python
artifact = ConditionalRQS.from_pretrained(
    "dide-ic/estiMINT",
    predictor="prev_y9",
    target="eir",
    revision="v1.0.0",
)
```

### 4. W&B sweeps

`estimint.v2.conf.sweeps.sweep.yaml` defines a Bayesian sweep over learning
rate, batch size, dropout, and the RQS architecture (`width`, `depth`,
`n_bins`, `rqs_bounds`, `mlp_residual`), minimising `val/loss`. Create a sweep,
then run agents using the ID returned by Weights & Biases:

```bash
uv run wandb sweep src/estimint/v2/conf/sweeps/sweep.yaml
uv run wandb agent <entity>/estimint-sweep/<sweep-id>
```

The sweep configuration currently targets `eir -> hbr_y9`; edit its final
Hydra overrides to sweep a different mapping.

### Configuration reference

- `src/estimint/v2/conf/train_config.yaml` defines data paths, split behavior,
  model architecture, optimization, checkpoint location, and W&B settings.
- `src/estimint/v2/conf/export_config.yaml` maps a training run's timestamp,
  checkpoint, scalers, and conformal JSON to an artifact directory.
- `src/estimint/v2/conf/sweeps/sweep.yaml` defines the W&B hyperparameter
  search space.

Run either entry point with `--help` to see the available Hydra options:

```bash
uv run python -m estimint.v2.train_base --help
uv run python -m estimint.v2.model_export --help
```

## Testing

```bash
uv sync --all-extras --dev
uv run pytest
```

The EIR-estimation and mosquito-delta tests download the published estiMINT
artifacts on first run. The full `run_scenarios` test is skipped unless the
`scenarios` extra is installed.

## License

MIT License