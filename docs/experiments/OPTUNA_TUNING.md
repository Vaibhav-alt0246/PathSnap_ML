# HistGradientBoosting Optuna tuning

A real-data Optuna study used a TPE sampler, seed `42`, the verified 24-feature
whitelist, all 80 trips, and the same 5-fold GroupKFold-by-trip structure. The
study completed 20 valid trials over the 1,514,674 candidate rows in 667.09
seconds.

Best trial:

```text
Top-1: 0.836964
Top-1 standard deviation: 0.037218
Mean log loss: 0.192618
learning_rate: 0.0321514
max_iter: 120
max_leaf_nodes: 26
min_samples_leaf: 70
l2_regularization: 1.631623
```

Comparison with the existing untuned baseline:

| Model | Top-1 | Log loss |
|---|---:|---:|
| Untuned HistGradientBoosting | 0.832225 | 0.181852 |
| Tuned best trial (20 trials) | 0.836964 | 0.192618 |

The 20-trial tuning improved Top-1 accuracy by approximately 0.0047, but
worsened log loss by approximately 0.0108. The untuned artifact remains the
safer probability-quality baseline for the future HMM until calibration and
route-level validation are completed.

Outputs are outside Git at `/tmp/pathsnap-optuna-20`:

- `optuna_trials.csv`
- `optuna_best.json`
- `hist_gradient_boosting_tuned.joblib`

The tuned artifact reloads successfully and contains the same 24-feature order.
