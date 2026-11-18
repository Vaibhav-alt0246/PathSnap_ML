# Development workflow

All work starts from the latest `main`. Branch ownership is `member-a-data`, `member-b-ml`, and `member-c-integration`. Push feature branches and open pull requests into `main`; keep commits focused and never force-push or rewrite shared history.

Member A owns data, projection, candidates, labels, and features. Member B owns models, validation, comparison, metrics, and tuning. Member C owns graph utilities, HMM/Viterbi, inference, API/UI, and integration tests.
