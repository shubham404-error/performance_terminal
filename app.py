from __future__ import annotations

import base64
import io
import time
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Performance Terminal", page_icon="📊", layout="wide")

# -----------------------------------------------------------------------------
# Master database schema. One row = one frozen signal occurrence.
# -----------------------------------------------------------------------------
DATABASE_COLUMNS = [
    "CohortDate", "SignalType", "Symbol", "Company", "YahooSymbol",
    "Universe", "Setup", "Score", "EntryPrice", "ImportedAt",
    "LastUpdated", "CurrentPrice", "TrailingProfitPct",
    "Return1D", "Return1W", "Return1M", "Return3M", "Return6M", "Return1Y",
    "Status",
]

GITHUB_DEFAULT_PATH = "performance_database.csv"
GITHUB_API = "https://api.github.com"

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def empty_database() -> pd.DataFrame:
    return pd.DataFrame(columns=DATABASE_COLUMNS)


def normalize_symbol(symbol) -> str:
    s = str(symbol).strip().upper()
    return s if s.endswith(".NS") else f"{s}.NS"


def safe_number(value):
    return pd.to_numeric(value, errors="coerce")


def pct(current, entry):
    if pd.isna(current) or pd.isna(entry) or float(entry) == 0:
        return np.nan
    return (float(current) / float(entry) - 1.0) * 100.0


def empty_if_missing(df: pd.DataFrame, column: str, default=np.nan):
    if column not in df.columns:
        df[column] = default


def normalize_database(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_database()
    result = df.copy()
    for col in DATABASE_COLUMNS:
        empty_if_missing(result, col)
    result = result[DATABASE_COLUMNS].copy()
    result["CohortDate"] = pd.to_datetime(result["CohortDate"], errors="coerce").dt.date.astype("string")
    for col in ["Score", "EntryPrice", "CurrentPrice", "TrailingProfitPct", "Return1D", "Return1W", "Return1M", "Return3M", "Return6M", "Return1Y"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.drop_duplicates(["CohortDate", "SignalType", "Symbol"], keep="last")
    return result.reset_index(drop=True)


def outcome_label(value) -> str:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "Pending"
    if value > 0:
        return "Winner"
    if value < 0:
        return "Loser"
    return "Neutral"


# -----------------------------------------------------------------------------
# GitHub master CSV storage
# Secrets expected:
# [github]
# token = "..."
# owner = "..."
# repo = "..."
# branch = "main"
# path = "performance_database.csv"
# -----------------------------------------------------------------------------
def github_config():
    try:
        gh = st.secrets.get("github", {})
    except Exception:
        gh = {}
    token = gh.get("token") or st.secrets.get("GITHUB_TOKEN", "")
    owner = gh.get("owner") or st.secrets.get("GITHUB_OWNER", "")
    repo = gh.get("repo") or st.secrets.get("GITHUB_REPO", "")
    branch = gh.get("branch") or st.secrets.get("GITHUB_BRANCH", "main")
    path = gh.get("path") or st.secrets.get("GITHUB_DB_PATH", GITHUB_DEFAULT_PATH)
    return {"token": token, "owner": owner, "repo": repo, "branch": branch, "path": path}


def github_ready(cfg) -> bool:
    return bool(cfg["token"] and cfg["owner"] and cfg["repo"])


def github_headers(cfg):
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {cfg['token']}",
        "X-GitHub-Api-Version": "2026-03-10",
    }


def github_url(cfg):
    return f"{GITHUB_API}/repos/{cfg['owner']}/{cfg['repo']}/contents/{cfg['path']}"


def load_database_from_github(cfg):
    if not github_ready(cfg):
        return empty_database(), None, "GitHub secrets are not configured."
    try:
        response = requests.get(
            github_url(cfg),
            headers=github_headers(cfg),
            params={"ref": cfg["branch"]},
            timeout=20,
        )
        if response.status_code == 404:
            return empty_database(), None, None
        response.raise_for_status()
        payload = response.json()
        raw = base64.b64decode(payload["content"].replace("\n", ""))
        df = pd.read_csv(io.BytesIO(raw))
        return normalize_database(df), payload.get("sha"), None
    except Exception as exc:
        return None, None, str(exc)


def merge_databases(remote: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """Merge by immutable signal identity, preferring local values for matching rows."""
    remote = normalize_database(remote)
    local = normalize_database(local)
    if remote.empty:
        return local
    if local.empty:
        return remote
    key = ["CohortDate", "SignalType", "Symbol"]
    combined = pd.concat([remote, local], ignore_index=True)
    return combined.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)


def save_database_to_github(cfg, database: pd.DataFrame, sha=None, commit_message="Update performance database"):
    if not github_ready(cfg):
        return False, None, "GitHub secrets are not configured."
    database = normalize_database(database)
    content = base64.b64encode(database.to_csv(index=False).encode("utf-8")).decode("ascii")
    payload = {"message": commit_message, "content": content, "branch": cfg["branch"]}
    if sha:
        payload["sha"] = sha
    try:
        response = requests.put(github_url(cfg), headers=github_headers(cfg), json=payload, timeout=30)
        if response.status_code in (200, 201):
            data = response.json()
            return True, data.get("content", {}).get("sha"), None
        if response.status_code == 409:
            return False, None, "GitHub conflict. Reload the database and retry."
        return False, None, f"GitHub save failed ({response.status_code}): {response.text[:300]}"
    except Exception as exc:
        return False, None, str(exc)


def persist_database(database: pd.DataFrame, message: str):
    cfg = github_config()
    sha = st.session_state.get("github_sha")
    ok, new_sha, err = save_database_to_github(cfg, database, sha=sha, commit_message=message)
    if not ok and err and "conflict" in err.lower():
        remote, remote_sha, load_err = load_database_from_github(cfg)
        if load_err is None:
            merged = merge_databases(remote, database)
            ok, new_sha, err = save_database_to_github(cfg, merged, sha=remote_sha, commit_message=message + " after merge")
            if ok:
                st.session_state["database"] = merged
    if ok:
        st.session_state["github_sha"] = new_sha
    return ok, err


# -----------------------------------------------------------------------------
# Batch Yahoo Finance history. One symbol is fetched once, then reused for every
# cohort row for that symbol. 100 rows with repeated names do not mean 100 calls.
# -----------------------------------------------------------------------------
def _extract_batch_close(data: pd.DataFrame, tickers: list[str]) -> dict[str, pd.Series]:
    out = {}
    if data is None or data.empty:
        return out
    try:
        if isinstance(data.columns, pd.MultiIndex):
            level0 = set(data.columns.get_level_values(0))
            level1 = set(data.columns.get_level_values(1))
            if "Close" in level0:
                closes = data["Close"]
            elif "Close" in level1:
                closes = data.xs("Close", axis=1, level=1)
            else:
                return out
            for ticker in tickers:
                if ticker in closes.columns:
                    series = pd.to_numeric(closes[ticker], errors="coerce").dropna()
                    if not series.empty:
                        series.index = pd.to_datetime(series.index).tz_localize(None)
                        out[ticker] = series
        else:
            close = pd.to_numeric(data["Close"], errors="coerce").dropna()
            if not close.empty and len(tickers) == 1:
                close.index = pd.to_datetime(close.index).tz_localize(None)
                out[tickers[0]] = close
    except Exception:
        return out
    return out


def fetch_histories_batch(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, progress_callback=None) -> dict[str, pd.Series]:
    symbols = sorted({str(x) for x in symbols if str(x).strip() and str(x).lower() != "nan"})
    if not symbols:
        return {}
    histories: dict[str, pd.Series] = {}
    chunk_size = 40
    chunks = [symbols[i:i + chunk_size] for i in range(0, len(symbols), chunk_size)]
    for idx, chunk in enumerate(chunks, start=1):
        try:
            data = yf.download(
                tickers=chunk,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
            histories.update(_extract_batch_close(data, chunk))
        except Exception:
            pass
        if progress_callback:
            progress_callback(idx, len(chunks), f"Downloading batch {idx}/{len(chunks)}")
        time.sleep(0.15)

    # Retry only a bounded number of missing symbols individually.
    # The batch path is the normal path. A Yahoo outage should not turn one update
    # into an unbounded series of one-by-one network calls.
    MAX_INDIVIDUAL_RETRIES = 15
    missing = [s for s in symbols if s not in histories]
    retry_symbols = missing[:MAX_INDIVIDUAL_RETRIES]

    for idx, symbol in enumerate(retry_symbols, start=1):
        try:
            data = yf.download(
                symbol,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            histories.update(_extract_batch_close(data, [symbol]))
        except Exception:
            pass
        if progress_callback and retry_symbols:
            progress_callback(
                idx,
                len(retry_symbols),
                f"Retrying unavailable symbols {idx}/{len(retry_symbols)}",
            )

    if progress_callback and len(missing) > MAX_INDIVIDUAL_RETRIES:
        progress_callback(
            len(retry_symbols),
            len(retry_symbols),
            f"Skipped {len(missing) - MAX_INDIVIDUAL_RETRIES} additional individual retries",
        )

    return histories


def close_on_or_before(series: pd.Series, target) -> float:
    if series is None or series.empty:
        return np.nan
    target = pd.Timestamp(target).normalize()
    eligible = series.loc[series.index <= target]
    return float(eligible.iloc[-1]) if not eligible.empty else np.nan


def close_on_or_before_calendar(series: pd.Series, target) -> float:
    return close_on_or_before(series, target)


def target_date(cohort_date, horizon: str) -> pd.Timestamp:
    ts = pd.Timestamp(cohort_date).normalize()
    if horizon == "1D":
        return ts + pd.Timedelta(days=1)
    if horizon == "1W":
        return ts + pd.DateOffset(weeks=1)
    if horizon == "1M":
        return ts + pd.DateOffset(months=1)
    if horizon == "3M":
        return ts + pd.DateOffset(months=3)
    if horizon == "6M":
        return ts + pd.DateOffset(months=6)
    if horizon == "1Y":
        return ts + pd.DateOffset(years=1)
    raise ValueError(f"Unknown horizon: {horizon}")


def metrics_from_history(series: pd.Series, cohort_date) -> dict:
    result = {
        "EntryPrice": np.nan,
        "CurrentPrice": np.nan,
        "TrailingProfitPct": np.nan,
        "Return1D": np.nan,
        "Return1W": np.nan,
        "Return1M": np.nan,
        "Return3M": np.nan,
        "Return6M": np.nan,
        "Return1Y": np.nan,
        "Status": "Price history unavailable",
    }
    if series is None or series.empty:
        return result
    cohort_ts = pd.Timestamp(cohort_date).normalize()
    entry = close_on_or_before(series, cohort_ts)
    if pd.isna(entry):
        return result
    result["EntryPrice"] = entry
    latest = series.loc[series.index <= pd.Timestamp.today().normalize()]
    if not latest.empty:
        current = float(latest.iloc[-1])
        result["CurrentPrice"] = current
        result["TrailingProfitPct"] = pct(current, entry)
    horizon_map = {
        "Return1D": "1D", "Return1W": "1W", "Return1M": "1M",
        "Return3M": "3M", "Return6M": "6M", "Return1Y": "1Y",
    }
    today = pd.Timestamp.today().normalize()
    for col, horizon in horizon_map.items():
        target = target_date(cohort_ts, horizon)
        if target <= today:
            checkpoint = close_on_or_before_calendar(series, target)
            result[col] = pct(checkpoint, entry)
    result["Status"] = "Historical" if target_date(cohort_ts, "1Y") <= today else "Active"
    return result


# -----------------------------------------------------------------------------
# Import preparation
# -----------------------------------------------------------------------------
def detect_universe(filename: str) -> str:
    name = str(filename).lower()
    if "nifty50" in name or "nifty_50" in name:
        return "Nifty 50"
    if "nifty200" in name or "nifty_200" in name:
        return "Nifty 200"
    if "nifty500" in name or "nifty_500" in name:
        return "Nifty 500"
    if "total" in name:
        return "Nifty Total Market"
    return "Unknown"


def prepare_signal(raw: pd.DataFrame, cohort_date, universe: str, signal_type: str) -> pd.DataFrame:
    df = raw.copy()
    if "Symbol" not in df.columns:
        raise ValueError("CSV is missing Symbol.")
    score_source = "ConvergenceScore" if signal_type == "Confluence" else "Investor Conviction"
    if score_source not in df.columns:
        raise ValueError(f"{signal_type} CSV is missing {score_source}.")
    df["Score"] = safe_number(df[score_source])
    df = df.loc[df["Score"] >= 75].copy()
    df["SignalType"] = signal_type
    df["Company"] = df["Company"] if "Company" in df.columns else df["Symbol"]
    df["Setup"] = df["Setup"] if "Setup" in df.columns else pd.NA
    df["Universe"] = universe
    df["CohortDate"] = pd.Timestamp(cohort_date).date().isoformat()
    df["YahooSymbol"] = df["Symbol"].map(normalize_symbol)
    return df[["CohortDate", "SignalType", "Symbol", "Company", "YahooSymbol", "Universe", "Setup", "Score"]]


def append_cohort(database: pd.DataFrame, incoming: pd.DataFrame, histories: dict[str, pd.Series]):
    database = normalize_database(database)
    if incoming.empty:
        return database, 0
    key = ["CohortDate", "SignalType", "Symbol"]
    existing = set(database[key].astype(str).agg("|".join, axis=1)) if not database.empty else set()
    incoming = incoming.copy()
    incoming_key = incoming[key].astype(str).agg("|".join, axis=1)
    new_rows = incoming.loc[~incoming_key.isin(existing)].copy()
    if new_rows.empty:
        return database, 0
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    for row in new_rows.itertuples(index=False):
        record = row._asdict()
        metrics = metrics_from_history(histories.get(record["YahooSymbol"], pd.Series(dtype=float)), record["CohortDate"])
        record.update(metrics)
        record["ImportedAt"] = now
        record["LastUpdated"] = now
        records.append(record)
    add_df = normalize_database(pd.DataFrame(records))
    result = pd.concat([database, add_df], ignore_index=True)
    result = normalize_database(result)
    return result, len(add_df)


def update_database(database: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """Refresh only cohorts whose return windows can still change.

    Completed 1-year cohorts are frozen. Active rows share one batch price download
    per unique Yahoo symbol, and the fetched series is reused across repeated signals.
    """
    result = normalize_database(database)
    if result.empty:
        return result

    today = pd.Timestamp.today().normalize()
    cohort_dates = pd.to_datetime(result["CohortDate"], errors="coerce").dt.normalize()
    active_mask = cohort_dates.notna() & (
        cohort_dates + pd.DateOffset(years=1) > today
    )

    active_indices = result.index[active_mask]
    if len(active_indices) == 0:
        if progress_callback:
            progress_callback(1, 1, "All cohorts are historical. No refresh needed.")
        return result

    active_rows = result.loc[active_indices].copy()
    symbols = (
        active_rows["YahooSymbol"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.strip().ne("") & s.str.lower().ne("nan")]
        .unique()
        .tolist()
    )

    if not symbols:
        if progress_callback:
            progress_callback(1, 1, "No valid active Yahoo symbols to refresh.")
        return result

    min_date = pd.to_datetime(
        active_rows["CohortDate"], errors="coerce"
    ).min()

    if pd.isna(min_date):
        return result

    start = pd.Timestamp(min_date).normalize() - pd.Timedelta(days=14)
    histories = fetch_histories_batch(
        symbols,
        start,
        today,
        progress_callback,
    )

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_records = []

    # Build updates first, then assign them in one batch instead of repeated .at writes.
    for pos, (idx, row) in enumerate(active_rows.iterrows(), start=1):
        series = histories.get(
            str(row["YahooSymbol"]),
            pd.Series(dtype=float),
        )
        metrics = metrics_from_history(series, row["CohortDate"])
        metrics["LastUpdated"] = now
        metrics["_index"] = idx
        metric_records.append(metrics)

        if progress_callback:
            progress_callback(
                pos,
                len(active_rows),
                f"Applying prices {pos}/{len(active_rows)}",
            )

    if metric_records:
        updates = pd.DataFrame(metric_records).set_index("_index")
        update_columns = [
            col for col in updates.columns
            if col in result.columns
        ]
        result.loc[updates.index, update_columns] = updates[update_columns]

    return normalize_database(result)


# -----------------------------------------------------------------------------
# Derived views
# -----------------------------------------------------------------------------
def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (cohort_date, signal_type), group in df.groupby(["CohortDate", "SignalType"], dropna=False):
        trailing = safe_number(group["TrailingProfitPct"])
        valid = trailing.dropna()
        best_idx = trailing.idxmax() if not valid.empty else None
        worst_idx = trailing.idxmin() if not valid.empty else None
        rows.append({
            "Cohort Date": cohort_date,
            "Signal Type": signal_type,
            "Signals": len(group),
            "Tracked": len(valid),
            "Trailing Avg %": valid.mean() if not valid.empty else np.nan,
            "Win Rate %": valid.gt(0).mean() * 100 if not valid.empty else np.nan,
            "1D %": safe_number(group["Return1D"]).mean(),
            "1W %": safe_number(group["Return1W"]).mean(),
            "1M %": safe_number(group["Return1M"]).mean(),
            "3M %": safe_number(group["Return3M"]).mean(),
            "Best": f"{group.loc[best_idx, 'Symbol']} {trailing.loc[best_idx]:+.2f}%" if best_idx is not None else "Pending",
            "Worst": f"{group.loc[worst_idx, 'Symbol']} {trailing.loc[worst_idx]:+.2f}%" if worst_idx is not None else "Pending",
        })
    return pd.DataFrame(rows).sort_values(["Cohort Date", "Signal Type"], ascending=[False, True])


def consolidated_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Consecutive repeated appearances become one trade episode. A gap over four calendar days starts a new episode."""
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["_date"] = pd.to_datetime(work["CohortDate"], errors="coerce")
    work = work.sort_values(["Symbol", "SignalType", "Universe", "_date"])
    rows = []
    for keys, group in work.groupby(["Symbol", "SignalType", "Universe"], dropna=False):
        group = group.sort_values("_date").copy()
        episode = 0
        previous = None
        episodes = []
        for idx, row in group.iterrows():
            if previous is not None and pd.notna(row["_date"]) and (row["_date"] - previous).days > 4:
                episode += 1
            episodes.append(episode)
            previous = row["_date"]
        group["_episode"] = episodes
        for _, ep in group.groupby("_episode"):
            ep = ep.sort_values("_date")
            first = ep.iloc[0]
            latest = ep.iloc[-1]
            entry = safe_number(pd.Series([first["EntryPrice"]])).iloc[0]
            current = safe_number(ep["CurrentPrice"]).dropna()
            current_price = float(current.iloc[-1]) if not current.empty else np.nan
            trailing = pct(current_price, entry)
            rows.append({
                "Symbol": first["Symbol"],
                "Company": first["Company"],
                "Signal Type": first["SignalType"],
                "Universe": first["Universe"],
                "Setup": first["Setup"],
                "First Signal": first["CohortDate"],
                "Latest Signal": latest["CohortDate"],
                "Signal Appearances": len(ep),
                "Entry Price": entry,
                "Current Price": current_price,
                "Trailing Profit %": trailing,
                "Outcome": outcome_label(trailing),
            })
    return pd.DataFrame(rows).sort_values(["First Signal", "Symbol"], ascending=[False, True]).reset_index(drop=True)


def build_whatsapp_update(detail: pd.DataFrame, cohort_date, signal_type: str) -> str:
    if detail.empty:
        return "No tracked signals available for this cohort."
    work = detail.copy()
    work["_ret"] = safe_number(work["TrailingProfitPct"])
    work = work.sort_values("_ret", ascending=False, na_position="last")
    tracked = work["_ret"].dropna()
    wins = int((tracked > 0).sum())
    losses = int((tracked < 0).sum())
    neutral = int((tracked == 0).sum())
    avg = tracked.mean() if not tracked.empty else np.nan
    win_rate = wins / len(tracked) * 100 if not tracked.empty else np.nan
    label = pd.to_datetime(cohort_date).strftime("%d %b %Y")
    lines = [
        f"📊 *{signal_type} Performance Update*",
        f"📅 Cohort: {label}",
        "",
        f"🎯 Signals: {len(work)}",
        f"📈 Trailing Average: {avg:+.2f}%" if pd.notna(avg) else "📈 Trailing Average: Pending",
        f"🟢 Win Rate: {win_rate:.1f}%" if pd.notna(win_rate) else "🟢 Win Rate: Pending",
        f"🟢 Winners: {wins}  |  🔴 Losers: {losses}  |  ⚪ Neutral: {neutral}",
        "",
        "*Running Signals*",
    ]
    for row in work.itertuples(index=False):
        symbol = str(getattr(row, "Symbol", "")).strip()
        ret = getattr(row, "_ret", np.nan)
        setup = str(getattr(row, "Setup", "")).strip()
        setup_text = f" | {setup}" if setup and setup.lower() != "nan" else ""
        if pd.notna(ret):
            icon = "🟢" if ret > 0 else ("🔴" if ret < 0 else "⚪")
            lines.append(f"{icon} *{symbol}*{setup_text}: {ret:+.2f}%")
        else:
            lines.append(f"⏳ *{symbol}*: Pending")
    # Keep the WhatsApp update compact: show only the best and worst tracked performers.
    # The full signal-level detail remains available in Raw Signal Performance.
    valid_rows = work.loc[work["_ret"].notna()].copy()

    if not valid_rows.empty:
        best = valid_rows.iloc[0]
        worst = valid_rows.iloc[-1]

        lines.extend([
            "",
            "*Best Performer*",
            f"🟢 *{best['Symbol']}*: {best['_ret']:+.2f}%",
            "",
            "*Worst Performer*",
            f"🔴 *{worst['Symbol']}*: {worst['_ret']:+.2f}%" if worst["_ret"] < 0 else f"⚪ *{worst['Symbol']}*: {worst['_ret']:+.2f}%",
        ])
    else:
        lines.extend(["", "⏳ Individual performance: Pending"])

    lines.extend(["", "_Research tracking only. Not investment advice._"])
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Initial load
# -----------------------------------------------------------------------------
if "database" not in st.session_state:
    cfg = github_config()
    db, sha, err = load_database_from_github(cfg)
    if err and github_ready(cfg):
        st.session_state["database"] = empty_database()
        st.session_state["github_load_error"] = err
    else:
        st.session_state["database"] = db if db is not None else empty_database()
        st.session_state["github_sha"] = sha

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
# Bump this whenever performance is refreshed so the WhatsApp text area gets a
# fresh Streamlit widget identity instead of retaining stale text.
st.session_state.setdefault("performance_update_version", 0)

st.title("Performance Terminal")
st.caption("Private cohort tracking with one persistent GitHub-backed master CSV.")

with st.expander("Methodology", expanded=False):
    st.markdown(
        """
- Every imported row is a frozen signal observation. The same stock can legitimately appear in multiple cohorts.
- Standard checkpoints use calendar conventions: **1D, 1W, 1M, 3M, 6M and 1Y**.
- If a checkpoint date is a weekend, NSE holiday or other non-trading date, the most recent available market close on or before that calendar date is used.
- **Trailing Profit** is the return from the frozen entry price to the latest available close and updates whenever performance is refreshed.
- The **Consolidated Trades** view is derived from the master CSV. Consecutive appearances of the same symbol and signal are treated as one trade episode so repeated names do not artificially inflate the trade count.
        """
    )

with st.sidebar:
    st.header("Master Database")
    cfg = github_config()
    if github_ready(cfg):
        st.success(f"Connected to {cfg['owner']}/{cfg['repo']}")
        st.caption(f"File: {cfg['path']}")
        if st.button("Reload from GitHub", use_container_width=True):
            db, sha, err = load_database_from_github(cfg)
            if err:
                st.error(f"Reload failed: {err}")
            else:
                st.session_state["database"] = db
                st.session_state["github_sha"] = sha
                st.success("Master CSV reloaded.")
                st.rerun()
    else:
        st.warning("GitHub secrets are not configured.")
        st.code('''[github]\ntoken = "YOUR_FINE_GRAINED_TOKEN"\nowner = "YOUR_GITHUB_USERNAME"\nrepo = "YOUR_PRIVATE_REPO"\nbranch = "main"\npath = "performance_database.csv"''', language="toml")

    st.divider()
    manual = st.file_uploader("Emergency local database load", type=["csv"])
    if manual is not None:
        try:
            st.session_state["database"] = normalize_database(pd.read_csv(manual))
            st.success("Loaded locally. Save to GitHub to make it persistent.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")


tab_import, tab_dashboard, tab_detail, tab_consolidated, tab_whatsapp, tab_export = st.tabs([
    "Import Cohort", "Dashboard", "Signal Detail", "Consolidated Trades", "WhatsApp Update", "Export"
])

with tab_import:
    st.subheader("Freeze a new cohort")
    c1, c2 = st.columns(2)
    with c1:
        cohort_date = st.date_input("Scan / cohort date", value=date.today())
    with c2:
        universe = st.selectbox("Universe", ["Nifty 50", "Nifty 200", "Nifty 500", "Nifty Total Market", "Unknown"], index=3)
    col1, col2 = st.columns(2)
    with col1:
        confluence_file = st.file_uploader("Confluence CSV", type=["csv"], key="confluence_upload")
    with col2:
        final_file = st.file_uploader("Final Buy List CSV", type=["csv"], key="final_upload")

    if st.button("Freeze Uploaded Signals", type="primary", use_container_width=True):
        candidates, errors = [], []
        for uploaded, signal_type in [(confluence_file, "Confluence"), (final_file, "Final Buy List")]:
            if uploaded is None:
                continue
            try:
                raw = pd.read_csv(uploaded)
                use_universe = universe if universe != "Unknown" else detect_universe(uploaded.name)
                candidates.append(prepare_signal(raw, cohort_date, use_universe, signal_type))
            except Exception as exc:
                errors.append(f"{signal_type}: {exc}")
        for err in errors:
            st.error(err)
        if candidates:
            incoming = pd.concat(candidates, ignore_index=True)
            if incoming.empty:
                st.warning("No uploaded signals cleared the 75 score threshold.")
            else:
                symbols = incoming["YahooSymbol"].unique().tolist()
                start = pd.Timestamp(cohort_date) - pd.Timedelta(days=14)
                end = pd.Timestamp.today().normalize()
                with st.spinner("Fetching entry history in batches..."):
                    histories = fetch_histories_batch(symbols, start, end)
                    updated, added = append_cohort(st.session_state["database"], incoming, histories)
                st.session_state["database"] = updated
                if added:
                    ok, err = persist_database(updated, f"Freeze cohort {pd.Timestamp(cohort_date).date()}")
                    if ok:
                        st.success(f"Frozen and saved {added} new signal(s).")
                    else:
                        st.error(f"Signals are in this session but GitHub save failed: {err}")
                else:
                    st.info("All uploaded signals already exist in the master database.")
                st.dataframe(incoming.sort_values(["SignalType", "Score"], ascending=[True, False]), use_container_width=True, hide_index=True)

with tab_dashboard:
    db = normalize_database(st.session_state["database"])
    st.subheader("Performance Dashboard")
    if db.empty:
        st.info("No cohorts yet.")
    else:
        trailing = safe_number(db["TrailingProfitPct"])
        tracked = trailing.dropna()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Signal Records", f"{len(db):,}")
        c2.metric("Unique Cohorts", f"{db[['CohortDate', 'SignalType']].drop_duplicates().shape[0]:,}")
        c3.metric("Trailing Avg", f"{tracked.mean():+.2f}%" if not tracked.empty else "Pending")
        c4.metric("Winners", f"{int((tracked > 0).sum()):,}")
        c5.metric("Win Rate", f"{tracked.gt(0).mean() * 100:.1f}%" if not tracked.empty else "Pending")

        if st.button("Update Performance and Save", type="primary"):
            progress = st.progress(0, text="Preparing price update...")
            def callback(done, total, text):
                progress.progress(min(done / max(total, 1), 1.0), text=text)
            updated = update_database(db, callback)
            st.session_state["database"] = updated

            # Streamlit preserves widget state across reruns. Incrementing this
            # version gives the WhatsApp text area a new key, so it reflects the
            # freshly calculated performance instead of an older Pending message.
            st.session_state["performance_update_version"] += 1

            ok, err = persist_database(updated, "Daily performance refresh")
            progress.empty()
            if ok:
                st.success("Performance refreshed and master CSV saved.")
            else:
                st.error(f"Performance refreshed locally, but GitHub save failed: {err}")

        summary = cohort_summary(st.session_state["database"])
        st.divider()
        st.subheader("Cohort History")
        st.dataframe(summary.style.format({
            "Trailing Avg %": "{:+.2f}%", "Win Rate %": "{:.1f}%", "1D %": "{:+.2f}%", "1W %": "{:+.2f}%", "1M %": "{:+.2f}%", "3M %": "{:+.2f}%"
        }, na_rep="Pending"), use_container_width=True, hide_index=True)

with tab_detail:
    db = normalize_database(st.session_state["database"])
    st.subheader("Raw Signal Performance")
    st.caption("Every cohort occurrence remains separate here. Repeated names are intentional signal observations.")
    if db.empty:
        st.info("No tracked signals yet.")
    else:
        cohorts = db[["CohortDate", "SignalType"]].drop_duplicates().sort_values(["CohortDate", "SignalType"], ascending=[False, True]).reset_index(drop=True)
        labels = [f"{r.CohortDate} . {r.SignalType}" for r in cohorts.itertuples(index=False)]
        selected = st.selectbox("Select cohort", labels)
        row = cohorts.iloc[labels.index(selected)]
        detail = db[(db["CohortDate"] == row["CohortDate"]) & (db["SignalType"] == row["SignalType"])].copy()
        display_cols = ["Symbol", "Company", "Universe", "Setup", "Score", "EntryPrice", "CurrentPrice", "TrailingProfitPct", "Return1D", "Return1W", "Return1M", "Return3M", "Return6M", "Return1Y", "Status"]
        display = detail[display_cols].sort_values("TrailingProfitPct", ascending=False, na_position="last")
        st.dataframe(display.style.format({
            "Score": "{:.1f}", "EntryPrice": "₹{:,.2f}", "CurrentPrice": "₹{:,.2f}", "TrailingProfitPct": "{:+.2f}%", "Return1D": "{:+.2f}%", "Return1W": "{:+.2f}%", "Return1M": "{:+.2f}%", "Return3M": "{:+.2f}%", "Return6M": "{:+.2f}%", "Return1Y": "{:+.2f}%"
        }, na_rep="Pending"), use_container_width=True, hide_index=True)

with tab_consolidated:
    db = normalize_database(st.session_state["database"])
    st.subheader("Consolidated Trades")
    st.caption("Consecutive repeated appearances of the same stock and signal are consolidated into one trade episode.")
    trades = consolidated_trades(db)
    if trades.empty:
        st.info("No consolidated trades yet.")
    else:
        selected_outcomes = st.multiselect("Trade outcome", ["Winner", "Loser", "Neutral", "Pending"], default=["Winner", "Loser", "Neutral", "Pending"])
        filtered = trades[trades["Outcome"].isin(selected_outcomes)].copy()
        a, b, c, d = st.columns(4)
        a.metric("Consolidated Trades", len(trades))
        b.metric("Winners", int((trades["Outcome"] == "Winner").sum()))
        c.metric("Losers", int((trades["Outcome"] == "Loser").sum()))
        valid = safe_number(trades["Trailing Profit %"]).dropna()
        d.metric("Trade Win Rate", f"{valid.gt(0).mean() * 100:.1f}%" if not valid.empty else "Pending")
        st.dataframe(filtered.style.format({"Entry Price": "₹{:,.2f}", "Current Price": "₹{:,.2f}", "Trailing Profit %": "{:+.2f}%"}, na_rep="Pending"), use_container_width=True, hide_index=True)

with tab_whatsapp:
    db = normalize_database(st.session_state["database"])
    st.subheader("WhatsApp Community Update")
    if db.empty:
        st.info("No tracked cohorts yet.")
    else:
        options = db[["CohortDate", "SignalType"]].drop_duplicates().sort_values(["CohortDate", "SignalType"], ascending=[False, True]).reset_index(drop=True)
        labels = [f"{r.CohortDate} . {r.SignalType}" for r in options.itertuples(index=False)]
        selected = st.selectbox("Choose cohort", labels, key="whatsapp_cohort")
        row = options.iloc[labels.index(selected)]
        detail = db[(db["CohortDate"] == row["CohortDate"]) & (db["SignalType"] == row["SignalType"])].copy()
        message = build_whatsapp_update(detail, row["CohortDate"], row["SignalType"])
        whatsapp_key = f"whatsapp_message_{st.session_state['performance_update_version']}"
        st.text_area(
            "Copy and paste into WhatsApp",
            value=message,
            key=whatsapp_key,
            height=min(max(320, 32 * (len(message.splitlines()) + 2)), 800),
        )

with tab_export:
    db = normalize_database(st.session_state["database"])
    st.subheader("Export")
    st.download_button("Download Master CSV", db.to_csv(index=False).encode("utf-8"), "performance_database.csv", "text/csv", use_container_width=True)
    summary = cohort_summary(db)
    st.download_button("Download Cohort Summary", summary.to_csv(index=False).encode("utf-8"), "cohort_performance_summary.csv", "text/csv", use_container_width=True)
    trades = consolidated_trades(db)
    st.download_button("Download Consolidated Trades", trades.to_csv(index=False).encode("utf-8"), "consolidated_trades.csv", "text/csv", use_container_width=True)

st.caption("Research tracking only. Performance uses adjusted closing prices from Yahoo Finance. Standard checkpoints follow calendar-period conventions, while Trailing Profit updates to the latest available close.")
