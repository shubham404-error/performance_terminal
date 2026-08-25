# Performance Terminal

A private Streamlit performance tracker for exported Nifty Market Terminal signals.

## What it does

- Imports Confluence and Final Buy List CSV exports.
- Applies a minimum score threshold of 75.
- Freezes the cohort date, signal metadata, and entry price at import.
- Prevents duplicate cohort imports.
- Tracks current, 1D, 5D, 10D, and 20D returns using trading sessions.
- Handles weekends and market holidays by using the latest available close on or before the cohort date.
- Keeps the original entry price immutable during performance updates.
- Supports cohort summaries and individual stock performance views.
- Generates a WhatsApp-ready performance update.
- Uses a portable Performance Database CSV, so no GitHub write access or external database is required.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a new GitHub repository.
2. Upload the contents of this folder.
3. Create a new Streamlit Community Cloud app.
4. Select the repository and set the main file path to:

```text
app.py
```

## Workflow

1. Export the Confluence CSV and/or Final Buy List CSV from the scanner.
2. Open Performance Terminal.
3. Choose the scan date and upload the export files.
4. Freeze the qualifying signals.
5. Download the Performance Database CSV.
6. On the next update, upload that database again.
7. Click Update Performance.
8. Download the refreshed database and copy the WhatsApp update if needed.

The app is intentionally portable and does not write to the public scanner repository.

## Important notes

Returns depend on market history availability from the data provider. If data is unavailable or insufficient for a return horizon, the app keeps that horizon unavailable rather than fabricating a value.

This tool is for research and tracking. It is not investment advice.
