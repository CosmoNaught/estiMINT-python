# models/prevalence

Prevalence → EIR model — the default estiMINT emulator.

Predicts **EIR** from year-9 prevalence (`prev_y9`) + 6 interventions. Bundled as
`prevalence` → `estiMINT_model.pkl`.

```bash
python models/prevalence/prepare.py   # datasets source -> training.parquet (prev_y9 >= 0.01)
python models/prevalence/train.py      # -> estiMINT_model.pkl, eir_xgb_FINAL.model, metrics/, plots/
```

Deployed copy lives in `src/estimint/data/estiMINT_model.pkl`.
