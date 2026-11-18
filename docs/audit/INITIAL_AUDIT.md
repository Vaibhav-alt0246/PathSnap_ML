# Initial engineering audit

Verified: graph history and tests are present; Member A and Member B branches exist remotely; Member A's Parquet is currently an unresolved Git LFS pointer; Member A uses `lon/lat/u/v` while the canonical integration contract uses `longitude/latitude/from_node/to_node`; HMM and real model artifacts were absent.

Blockers: raw training data and the actual Parquet object are unavailable; real five-model training cannot be honestly run; production inference cannot be enabled without a trained artifact; Member A reports contain inconsistent recall/prevalence figures.

No student-test reference routes were read. No real model metrics are claimed.
