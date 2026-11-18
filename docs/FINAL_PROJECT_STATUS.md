# Current project status

Implemented and tested: the Member A schema adapter, directed MultiDiGraph construction,
deterministic shortest-path arc selection, log-space HMM decoding with directed
connectivity, honest API readiness/trip listing, real 80-trip candidate validation,
five-model grouped comparison, a 20-trial Optuna study, the baseline/demo artifacts,
and the expanded test suite.

Completed: a wired GPS-to-road inference command through candidate generation,
feature engineering, classifier, and HMM; live API prediction; and Folium map
generation were verified on real training trip `00000033`. Still incomplete are
the complete 80-trip out-of-fold HMM aggregate evaluation using the optimized sparse
decoder. All 80 trips were validated exactly once, 16 per fold. The original student
score `0.000459` was audited and shown to be unsuitable positional comparison; the
corrected independent audit reports mean arc-set recall `0.919452` and LCS recall
`0.913063`. All 10 student-test predictions were generated and separately evaluated,
but their metrics were not used for tuning. The OOF HMM improved continuity but did
not improve the current route-membership diagnostic. Raw trip files are
available in `/Users/vaby/Downloads/Student folder/train`, but no `.route` files were
read for model selection or demonstration. These items remain assignment-level
integration work rather than evidence that the full system is complete.
