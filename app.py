import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Performance Terminal",
    page_icon="📊",
    layout="wide",
)

DATABASE_COLUMNS = [
    "CohortDate", "SignalType", "Symbol", "Company", "YahooSymbol",
    "Universe", "Setup", "Score", "EntryPrice",
    "ImportedAt", "LastUpdated", "CurrentPrice", "CurrentReturnPct",
    "Return1D", "Return5D", "Return10D", "Return20D",
    "TradingDaysElapsed", "Status",
]

def empty_database():
    return pd.DataFrame(columns=DATABASE_COLUMNS)

def normalize_symbol(symbol):
    s = str(symbol).strip().upper()
    return s if s.endswith(".NS") else f"{s}.NS"

def safe_number(series):
    return pd.to_numeric(series, errors="coerce")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(yahoo_symbol, start, end):
    try:
        data = yf.download(
            yahoo_symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = pd.to_numeric(close, errors="coerce").dropna()
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close
    except Exception:
        return pd.Series(dtype=float)

def close_on_or_before(yahoo_symbol, target_date):
    target = pd.Timestamp(target_date).normalize()
    start = (target - pd.Timedelta(days=14)).date().isoformat()
    end = (target + pd.Timedelta(days=2)).date().isoformat()
    close = fetch_history(yahoo_symbol, start, end)
    if close.empty:
        return np.nan
    eligible = close.loc[close.index <= target]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[-1])

def returns_from_history(yahoo_symbol, cohort_date):
    cohort_ts = pd.Timestamp(cohort_date).normalize()
    start = (cohort_ts - pd.Timedelta(days=10)).date().isoformat()
    end = (date.today() + timedelta(days=2)).isoformat()
    close = fetch_history(yahoo_symbol, start, end)

    result = {
        "EntryPrice": np.nan,
        "CurrentPrice": np.nan,
        "CurrentReturnPct": np.nan,
        "Return1D": np.nan,
        "Return5D": np.nan,
        "Return10D": np.nan,
        "Return20D": np.nan,
        "TradingDaysElapsed": 0,
    }
    if close.empty:
        return result

    eligible = close.loc[close.index <= cohort_ts]
    if eligible.empty:
        return result

    entry_idx = eligible.index[-1]
    entry_price = float(eligible.iloc[-1])
    future = close.loc[close.index > entry_idx]

    result["EntryPrice"] = entry_price
    result["TradingDaysElapsed"] = int(len(future))

    if not future.empty:
        current = float(future.iloc[-1])
        result["CurrentPrice"] = current
        result["CurrentReturnPct"] = (current / entry_price - 1) * 100

    for horizon, column in [(1, "Return1D"), (5, "Return5D"), (10, "Return10D"), (20, "Return20D")]:
        if len(future) >= horizon:
            price = float(future.iloc[horizon - 1])
            result[column] = (price / entry_price - 1) * 100

    return result

def detect_universe(filename):
    name = filename.lower()
    if "nifty50" in name or "nifty_50" in name:
        return "Nifty 50"
    if "nifty200" in name or "nifty_200" in name:
        return "Nifty 200"
    if "nifty500" in name or "nifty_500" in name:
        return "Nifty 500"
    if "total" in name:
        return "Nifty Total Market"
    return "Unknown"

def prepare_confluence(raw, cohort_date, universe):
    required = {"Symbol", "ConvergenceScore"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Confluence CSV is missing: {', '.join(sorted(missing))}")

    df = raw.copy()
    df["Score"] = safe_number(df["ConvergenceScore"])
    df = df.loc[df["Score"] >= 75].copy()
    df["SignalType"] = "Confluence"
    df["Company"] = df.get("Company", df["Symbol"])
    df["Setup"] = df.get("Setup", pd.NA)
    df["Universe"] = universe
    df["CohortDate"] = pd.Timestamp(cohort_date).date().isoformat()
    df["YahooSymbol"] = df["Symbol"].map(normalize_symbol)
    return df[["CohortDate","SignalType","Symbol","Company","YahooSymbol","Universe","Setup","Score"]]

def prepare_final_buy(raw, cohort_date, universe):
    required = {"Symbol", "Investor Conviction"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Final Buy List CSV is missing: {', '.join(sorted(missing))}")

    df = raw.copy()
    df["Score"] = safe_number(df["Investor Conviction"])
    df = df.loc[df["Score"] >= 75].copy()
    df["SignalType"] = "Final Buy List"
    df["Company"] = df.get("Company", df["Symbol"])
    df["Setup"] = df.get("Setup", pd.NA)
    df["Universe"] = universe
    df["CohortDate"] = pd.Timestamp(cohort_date).date().isoformat()
    df["YahooSymbol"] = df["Symbol"].map(normalize_symbol)
    return df[["CohortDate","SignalType","Symbol","Company","YahooSymbol","Universe","Setup","Score"]]

def append_cohort(database, incoming):
    if incoming.empty:
        return database.copy(), 0
    result = database.copy()
    if result.empty:
        result = empty_database()

    key_cols = ["CohortDate", "SignalType", "Symbol"]
    existing = set(
        result[key_cols].astype(str).agg("|".join, axis=1).tolist()
    ) if not result.empty else set()

    incoming = incoming.copy()
    keys = incoming[key_cols].astype(str).agg("|".join, axis=1)
    new_rows = incoming.loc[~keys.isin(existing)].copy()

    if new_rows.empty:
        return result, 0

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    for col in ["EntryPrice","LastUpdated","CurrentPrice","CurrentReturnPct",
                "Return1D","Return5D","Return10D","Return20D","TradingDaysElapsed","Status"]:
        new_rows[col] = np.nan

    new_rows["ImportedAt"] = now
    new_rows["EntryPrice"] = [
        close_on_or_before(symbol, row_date)
        for symbol, row_date in zip(new_rows["YahooSymbol"], new_rows["CohortDate"])
    ]
    new_rows["Status"] = np.where(new_rows["EntryPrice"].notna(), "Active", "Entry price unavailable")
    new_rows["TradingDaysElapsed"] = 0

    result = pd.concat([result, new_rows[DATABASE_COLUMNS]], ignore_index=True)
    return result, len(new_rows)

def update_database(database, progress_callback=None):
    result = database.copy()
    if result.empty:
        return result

    total = len(result)
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    for i, (_, row) in enumerate(result.iterrows(), start=1):
        metrics = returns_from_history(row["YahooSymbol"], row["CohortDate"])

        for key, value in metrics.items():
            result.at[row.name, key] = value

        if pd.notna(metrics["EntryPrice"]):
            result.at[row.name, "Status"] = (
                "Completed" if metrics["TradingDaysElapsed"] >= 20 else "Active"
            )
        else:
            result.at[row.name, "Status"] = "Price history unavailable"

        result.at[row.name, "LastUpdated"] = now

        if progress_callback:
            progress_callback(i, total)

    return result

def format_pct(value):
    if pd.isna(value):
        return "Pending"
    return f"{value:+.2f}%"

def cohort_summary(df):
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (cohort_date, signal_type), group in df.groupby(["CohortDate", "SignalType"], dropna=False):
        current = safe_number(group["CurrentReturnPct"])
        valid = current.dropna()
        best_idx = current.idxmax() if not valid.empty else None
        worst_idx = current.idxmin() if not valid.empty else None

        rows.append({
            "Cohort Date": cohort_date,
            "Signal Type": signal_type,
            "Signals": len(group),
            "Tracked": len(valid),
            "Current Avg %": valid.mean() if not valid.empty else np.nan,
            "Win Rate %": (valid.gt(0).mean() * 100) if not valid.empty else np.nan,
            "1D %": safe_number(group["Return1D"]).mean(),
            "5D %": safe_number(group["Return5D"]).mean(),
            "10D %": safe_number(group["Return10D"]).mean(),
            "20D %": safe_number(group["Return20D"]).mean(),
            "Best": (
                f"{group.loc[best_idx, 'Symbol']} {current.loc[best_idx]:+.2f}%"
                if best_idx is not None else "Pending"
            ),
            "Worst": (
                f"{group.loc[worst_idx, 'Symbol']} {current.loc[worst_idx]:+.2f}%"
                if worst_idx is not None else "Pending"
            ),
            "Status": "Completed" if (safe_number(group["TradingDaysElapsed"]) >= 20).all() else "Running",
        })

    return pd.DataFrame(rows).sort_values(["Cohort Date", "Signal Type"], ascending=[False, True])

# ---------------- UI ----------------

st.title("Performance Terminal")
st.caption(
    "Private cohort tracking for Nifty Market Terminal signals. "
    "Upload exported Confluence and Final Buy List CSVs, then track their performance."
)

with st.expander("How this works", expanded=False):
    st.markdown(
        """
1. Upload the exported **Confluence CSV** and/or **Final Buy List CSV**.
2. Select the date on which the scan was generated.
3. Only signals with a score of **75 or above** are imported.
4. The app freezes the signal date, score, setup and reference closing price.
5. Upload the downloaded Performance Database next time, so the app continues from the same history.
6. Click **Update Performance** to calculate 1D, 5D, 10D, 20D and current returns.

The Performance Database is intentionally portable. This version does not require GitHub, a database, or any changes to the live scanner.
        """
    )

if "database" not in st.session_state:
    st.session_state["database"] = empty_database()

with st.sidebar:
    st.header("Database")
    db_upload = st.file_uploader(
        "Load existing Performance Database",
        type=["csv"],
        key="database_upload",
    )
    if db_upload is not None:
        try:
            loaded = pd.read_csv(db_upload)
            missing = set(DATABASE_COLUMNS) - set(loaded.columns)
            if missing:
                st.error("This is not a compatible Performance Database CSV.")
            else:
                st.session_state["database"] = loaded[DATABASE_COLUMNS].copy()
                st.success(f"Loaded {len(loaded):,} tracked signals.")
        except Exception as exc:
            st.error(f"Could not load database: {exc}")

    if st.button("Start New Database", use_container_width=True):
        st.session_state["database"] = empty_database()
        st.rerun()

tab_import, tab_dashboard, tab_detail, tab_export = st.tabs(
    ["Import Cohort", "Dashboard", "Signal Detail", "Export"]
)

with tab_import:
    st.subheader("Freeze a new cohort")
    st.write("Upload one or both exports from the same market scan.")

    c1, c2 = st.columns(2)
    with c1:
        cohort_date = st.date_input("Scan / cohort date", value=date.today())
    with c2:
        universe = st.selectbox(
            "Universe",
            ["Nifty 50", "Nifty 200", "Nifty 500", "Nifty Total Market", "Unknown"],
            index=3,
        )

    col1, col2 = st.columns(2)
    with col1:
        confluence_file = st.file_uploader(
            "Confluence CSV",
            type=["csv"],
            key="confluence_upload",
        )
    with col2:
        final_file = st.file_uploader(
            "Final Buy List CSV",
            type=["csv"],
            key="final_upload",
        )

    if st.button("Freeze Uploaded Signals", type="primary", use_container_width=True):
        candidates = []
        errors = []

        if confluence_file is not None:
            try:
                raw = pd.read_csv(confluence_file)
                detected = detect_universe(confluence_file.name)
                use_universe = universe if universe != "Unknown" else detected
                candidates.append(prepare_confluence(raw, cohort_date, use_universe))
            except Exception as exc:
                errors.append(f"Confluence: {exc}")

        if final_file is not None:
            try:
                raw = pd.read_csv(final_file)
                detected = detect_universe(final_file.name)
                use_universe = universe if universe != "Unknown" else detected
                candidates.append(prepare_final_buy(raw, cohort_date, use_universe))
            except Exception as exc:
                errors.append(f"Final Buy List: {exc}")

        for error in errors:
            st.error(error)

        if candidates:
            incoming = pd.concat(candidates, ignore_index=True)
            if incoming.empty:
                st.warning("No uploaded signals cleared the 75 score threshold.")
            else:
                with st.spinner("Freezing entry prices from market history..."):
                    updated_db, added = append_cohort(
                        st.session_state["database"], incoming
                    )
                st.session_state["database"] = updated_db
                if added:
                    st.success(f"Frozen {added} new signal(s).")
                else:
                    st.info("All uploaded signals already exist in the database.")

                preview = incoming.sort_values(["SignalType", "Score"], ascending=[True, False])
                st.dataframe(preview, use_container_width=True, hide_index=True)

with tab_dashboard:
    db = st.session_state["database"]

    st.subheader("Performance Dashboard")

    if db.empty:
        st.info("No cohorts yet. Import your first Confluence or Final Buy List CSV.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Tracked Signals", f"{len(db):,}")
        a2 = db["Status"].eq("Active").sum()
        b.metric("Active", f"{int(a2):,}")
        c.metric("Completed", f"{int(db['Status'].eq('Completed').sum()):,}")
        d.metric("Cohorts", f"{db[['CohortDate','SignalType']].drop_duplicates().shape[0]:,}")

        if st.button("Update Performance", type="primary"):
            progress = st.progress(0, text="Updating price histories...")

            def callback(done, total):
                progress.progress(done / total, text=f"Updating {done}/{total} signals...")

            updated = update_database(db, callback)
            st.session_state["database"] = updated
            progress.empty()
            st.success("Performance updated.")

        summary = cohort_summary(st.session_state["database"])
        st.divider()
        st.subheader("Cohort History")
        if not summary.empty:
            st.dataframe(
                summary.style.format({
                    "Current Avg %": "{:+.2f}%",
                    "Win Rate %": "{:.1f}%",
                    "1D %": "{:+.2f}%",
                    "5D %": "{:+.2f}%",
                    "10D %": "{:+.2f}%",
                    "20D %": "{:+.2f}%",
                }, na_rep="Pending"),
                use_container_width=True,
                hide_index=True,
            )

with tab_detail:
    db = st.session_state["database"]
    st.subheader("Signal Detail")

    if db.empty:
        st.info("No tracked signals yet.")
    else:
        cohort_options = (
            db[["CohortDate", "SignalType"]]
            .drop_duplicates()
            .sort_values(["CohortDate", "SignalType"], ascending=[False, True])
        )
        labels = [
            f"{row.CohortDate} . {row.SignalType}"
            for row in cohort_options.itertuples(index=False)
        ]
        selected = st.selectbox("Select cohort", labels)
        selected_row = cohort_options.iloc[labels.index(selected)]

        detail = db.loc[
            (db["CohortDate"] == selected_row["CohortDate"])
            & (db["SignalType"] == selected_row["SignalType"])
        ].copy()

        display = detail[[
            "Symbol", "Company", "Universe", "Setup", "Score",
            "EntryPrice", "CurrentPrice", "CurrentReturnPct",
            "Return1D", "Return5D", "Return10D", "Return20D",
            "TradingDaysElapsed", "Status",
        ]].sort_values("CurrentReturnPct", ascending=False, na_position="last")

        st.dataframe(
            display.style.format({
                "Score": "{:.1f}",
                "EntryPrice": "₹{:,.2f}",
                "CurrentPrice": "₹{:,.2f}",
                "CurrentReturnPct": "{:+.2f}%",
                "Return1D": "{:+.2f}%",
                "Return5D": "{:+.2f}%",
                "Return10D": "{:+.2f}%",
                "Return20D": "{:+.2f}%",
            }, na_rep="Pending"),
            use_container_width=True,
            hide_index=True,
        )

        valid = safe_number(detail["CurrentReturnPct"]).dropna()
        if not valid.empty:
            best = detail.loc[safe_number(detail["CurrentReturnPct"]).idxmax()]
            worst = detail.loc[safe_number(detail["CurrentReturnPct"]).idxmin()]
            x, y, z = st.columns(3)
            x.metric("Current Basket Return", f"{valid.mean():+.2f}%")
            y.metric("Best", f"{best['Symbol']} {best['CurrentReturnPct']:+.2f}%")
            z.metric("Worst", f"{worst['Symbol']} {worst['CurrentReturnPct']:+.2f}%")

with tab_export:
    db = st.session_state["database"]

    st.subheader("Export")
    st.write("Download this file after importing or updating. Upload the same file next time to continue tracking.")

    st.download_button(
        "Download Performance Database",
        data=db.to_csv(index=False).encode("utf-8"),
        file_name="performance_terminal_database.csv",
        mime="text/csv",
        use_container_width=True,
    )

    summary = cohort_summary(db)
    st.download_button(
        "Download Cohort Summary",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="cohort_performance_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "Research tracking only. Returns use closing prices from the selected cohort date and subsequent trading sessions."
)
