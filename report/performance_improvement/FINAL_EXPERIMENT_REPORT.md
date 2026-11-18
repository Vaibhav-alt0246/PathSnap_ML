# Performance improvement report

The preserved baseline and 20-trial Optuna experiment were audited. The Optuna
study improved Top-1 from 0.832225 to 0.836964 but worsened mean log loss from
0.181852 to 0.192618, so the untuned baseline remains preferable for probability
quality and was not replaced.

Candidate diagnostics and feature permutation importance are provided in the CSV
files. The OOF HMM completed all 80 trips: classifier route-membership diagnostic
0.975598 versus HMM 0.555742, while continuity was 0.332611 versus 0.924116.
Thus HMM improves continuity but not the recorded membership diagnostic.

No evaluator or student-test data was used for model development. No reliable new
production improvement was demonstrated, so no model or feature configuration was
changed. Further candidate-radius experiments should be run as separate GroupKFold
studies before changing inference defaults.
