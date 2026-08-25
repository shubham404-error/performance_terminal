# Performance Terminal

Private cohort tracking for Nifty Market Terminal signals. The persistent store is one file: `performance_database.csv`.

## GitHub Secrets

```toml
[github]
token = "YOUR_FINE_GRAINED_TOKEN"
owner = "YOUR_GITHUB_USERNAME"
repo = "YOUR_PRIVATE_REPO"
branch = "main"
path = "performance_database.csv"
```

The token needs repository Contents read and write permission.

## Return methodology

- Entry: latest available close on or before the cohort date.
- 1D, 1W, 1M, 3M, 6M and 1Y: calendar checkpoints.
- If the checkpoint is non-trading, the latest available close on or before that date is used.
- Trailing Profit: latest available close versus frozen entry price.
- Raw Signal Performance keeps every signal occurrence. Consolidated Trades groups consecutive repeated appearances of the same symbol and signal into one trade episode.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
