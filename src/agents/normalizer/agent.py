"""
Normalizer node — pure Python, no LLM calls.

Reads all parquet files from state["data_dir"], normalizes them into a
canonical long-format prices.parquet, and writes data_manifest.json.

Canonical schema (long format — one row per date-ticker pair):
  date       datetime64[ns] — column, NOT index
  ticker     str
  open       float64
  high       float64
  low        float64
  close      float64
  adj_close  float64   (copy of close when not present in source)
  volume     float64   (NaN when not present in source)

Handles five input formats produced by OpenHands/yfinance/openbb:
  Case 1 — yfinance MultiIndex wide (most common multi-ticker download)
  Case 2 — yfinance flat wide (single-ticker download)
  Case 3 — one parquet per ticker (openbb style)
  Case 4 — already long format
  Case 5 — empty data_dir or no parquet files
"""
from __future__ import annotations

import json
import logging
import pathlib

import numpy as np
import pandas as pd

from src.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical column ordering
# ---------------------------------------------------------------------------

_CANONICAL_COLS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

# Case-insensitive mapping from source column names to canonical names
_COL_MAP: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    # adj_close variants
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjusted_close": "adj_close",
    "adjclose": "adj_close",
    "adjusted close": "adj_close",
    # volume variants
    "volume": "volume",
    "vol": "volume",
    # date variants (for long-format detection)
    "date": "date",
    "ticker": "ticker",
    "symbol": "ticker",
}


def _normalize_col_name(col: str) -> str | None:
    """Return canonical name for col, or None if not recognized."""
    return _COL_MAP.get(col.strip().lower())


# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------


def _is_multiindex_wide(df: pd.DataFrame) -> bool:
    """True for yfinance multi-ticker download: MultiIndex columns (field, ticker)."""
    return isinstance(df.columns, pd.MultiIndex)


def _is_long_format(df: pd.DataFrame) -> bool:
    """True when the frame already has date/ticker columns."""
    cols_lower = {c.lower() for c in df.columns}
    return ("ticker" in cols_lower or "symbol" in cols_lower) and (
        "date" in cols_lower or isinstance(df.index, pd.DatetimeIndex)
    )


def _is_flat_wide(df: pd.DataFrame) -> bool:
    """True for yfinance single-ticker flat wide (OHLCV columns, datetime index)."""
    cols_lower = {c.lower() for c in df.columns}
    return isinstance(df.index, pd.DatetimeIndex) and bool(
        {"open", "close"} & cols_lower
    ) and not _is_long_format(df)


# ---------------------------------------------------------------------------
# Converters — each returns a long-format DataFrame with canonical columns
# ---------------------------------------------------------------------------


def _from_multiindex_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yfinance MultiIndex wide format to long.

    df.columns is a MultiIndex like [("Open","AAPL"), ("High","AAPL"), ...]
    df.index is a DatetimeIndex.
    """
    # Normalize the level-0 field names
    df.columns = pd.MultiIndex.from_tuples(
        [
            (_normalize_col_name(str(field)) or str(field).lower(), str(ticker))
            for field, ticker in df.columns
        ]
    )

    # Stack: date → field pivot → long rows per (date, ticker)
    # After stack(level=1) the frame has MultiIndex (date, ticker) index and
    # columns = canonical field names.
    long = df.stack(level=1, future_stack=True).reset_index()

    # Rename the index columns produced by reset_index
    # LangGraph produces "level_0"/"level_1" or the actual index names
    rename = {}
    for col in long.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        if col_lower in ("date", "level_0") and "date" not in rename.values():
            rename[col_str] = "date"
        elif col_lower in ("ticker", "level_1", "symbol") and "ticker" not in rename.values():
            rename[col_str] = "ticker"
    long = long.rename(columns=rename)

    return long


def _from_flat_wide(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Convert single-ticker flat wide (DatetimeIndex, OHLCV columns) to long.
    ticker is derived from the filename.
    """
    long = df.reset_index()
    # The index column after reset could be named "Date", "date", "index", etc.
    rename = {}
    for col in long.columns:
        col_str = str(col)
        col_lower = col_str.lower()
        if col_lower in ("date", "datetime", "index") and "date" not in rename.values():
            rename[col_str] = "date"
        else:
            mapped = _normalize_col_name(col_str)
            if mapped:
                rename[col_str] = mapped
    long = long.rename(columns=rename)
    long["ticker"] = ticker
    return long


def _from_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an already-long frame: standardize column names and ensure
    date is a column (not index).
    """
    if isinstance(df.index, pd.DatetimeIndex) and "date" not in {c.lower() for c in df.columns}:
        df = df.reset_index()

    rename = {}
    for col in df.columns:
        mapped = _normalize_col_name(str(col))
        if mapped and mapped != str(col):
            rename[str(col)] = mapped
    return df.rename(columns=rename)


def _from_per_ticker(files: list[pathlib.Path]) -> pd.DataFrame:
    """
    Merge multiple single-ticker parquet files (one per ticker, openbb style).
    Each file: date as column or index, OHLCV columns.
    """
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            raw = pd.read_parquet(f)
        except Exception as exc:
            logger.warning("Skipping %s — read error: %s", f, exc)
            continue

        ticker = f.stem.upper()

        if isinstance(raw.index, pd.DatetimeIndex):
            chunk = _from_flat_wide(raw, ticker)
        elif _is_long_format(raw):
            chunk = _from_long(raw)
            if "ticker" not in chunk.columns:
                chunk["ticker"] = ticker
        else:
            # date may already be a column
            chunk = raw.copy()
            rename = {}
            for col in chunk.columns:
                mapped = _normalize_col_name(str(col))
                if mapped:
                    rename[str(col)] = mapped
            chunk = chunk.rename(columns=rename)
            if "ticker" not in chunk.columns:
                chunk["ticker"] = ticker

        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=_CANONICAL_COLS)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Post-processing — enforce types, fill missing cols, sort
# ---------------------------------------------------------------------------


def _enforce_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a long-format frame with at least date/ticker/close columns,
    ensure every canonical column exists with the right type.
    """
    if df.empty:
        return pd.DataFrame(columns=_CANONICAL_COLS)

    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
        # Drop timezone info — standardize to naive datetime
        if hasattr(df["date"].dtype, "tz") and df["date"].dtype.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    else:
        logger.warning("No 'date' column found after normalization — dropping frame")
        return pd.DataFrame(columns=_CANONICAL_COLS)

    # Ensure ticker column
    if "ticker" not in df.columns:
        df["ticker"] = "UNKNOWN"
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    # Numeric OHLCV columns
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # adj_close: use source if present, else copy close
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"].copy()
    else:
        df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
        # Fill remaining NaN adj_close with close (futures/indices)
        mask = df["adj_close"].isna()
        if mask.any():
            df.loc[mask, "adj_close"] = df.loc[mask, "close"]

    # volume: NaN when unavailable
    if "volume" not in df.columns:
        df["volume"] = np.nan
    else:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Drop rows where date is NaT (parse failures)
    df = df.dropna(subset=["date"])

    # Select and order canonical columns only
    df = df[_CANONICAL_COLS].copy()

    # Sort: ticker then date for reproducible output
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Main normalizer logic
# ---------------------------------------------------------------------------


def _build_empty_manifest() -> dict:
    return {
        "file": "prices.parquet",
        "tickers": [],
        "date_range": [],
        "row_count": 0,
        "columns": _CANONICAL_COLS,
    }


def _normalize_data_dir(data_dir: pathlib.Path) -> tuple[pd.DataFrame, dict]:
    """
    Discover format, normalize to canonical long DataFrame, return (df, manifest).
    """
    if not data_dir.exists():
        logger.warning("data_dir does not exist: %s", data_dir)
        return pd.DataFrame(columns=_CANONICAL_COLS), _build_empty_manifest()

    all_parquets = sorted(data_dir.glob("*.parquet"))
    # Exclude any previously-written prices.parquet to avoid self-contamination
    raw_parquets = [p for p in all_parquets if p.stem != "prices"]

    if not raw_parquets:
        # Check if prices.parquet itself already exists (idempotent re-run)
        prices_path = data_dir / "prices.parquet"
        if prices_path.exists():
            logger.info("prices.parquet already exists — re-reading for manifest")
            try:
                df = pd.read_parquet(prices_path)
                if _is_long_format(df):
                    df = _enforce_canonical(df)
                    return df, _make_manifest(df)
            except Exception as exc:
                logger.warning("Could not re-read prices.parquet: %s", exc)
        logger.warning("No parquet files found in %s", data_dir)
        return pd.DataFrame(columns=_CANONICAL_COLS), _build_empty_manifest()

    # --- Peek at first file to detect format ---
    try:
        first_df = pd.read_parquet(raw_parquets[0])
    except Exception as exc:
        logger.warning("Could not read first parquet %s: %s", raw_parquets[0], exc)
        return pd.DataFrame(columns=_CANONICAL_COLS), _build_empty_manifest()

    # Case 1 — MultiIndex wide (yfinance multi-ticker)
    if _is_multiindex_wide(first_df):
        logger.info("Detected format: yfinance MultiIndex wide")
        if len(raw_parquets) == 1:
            long = _from_multiindex_wide(first_df)
        else:
            # Multiple MultiIndex files — concatenate then convert
            frames = []
            for p in raw_parquets:
                try:
                    frames.append(pd.read_parquet(p))
                except Exception:
                    pass
            combined = pd.concat(frames, axis=0)
            # Rebuild MultiIndex properly after concat
            combined.columns = pd.MultiIndex.from_tuples(combined.columns)
            long = _from_multiindex_wide(combined)

        normalized = _enforce_canonical(long)

    # Case 4 — already long format (check before flat-wide since long may have DatetimeIndex)
    elif _is_long_format(first_df):
        logger.info("Detected format: long format (Case 4)")
        if len(raw_parquets) == 1:
            long = _from_long(first_df)
        else:
            frames = []
            for p in raw_parquets:
                try:
                    frames.append(_from_long(pd.read_parquet(p)))
                except Exception:
                    pass
            long = pd.concat(frames, ignore_index=True)
        normalized = _enforce_canonical(long)

    # Case 2 — flat wide single-ticker (single file, DatetimeIndex)
    elif _is_flat_wide(first_df) and len(raw_parquets) == 1:
        logger.info("Detected format: yfinance flat wide single-ticker (Case 2)")
        ticker = raw_parquets[0].stem.upper()
        long = _from_flat_wide(first_df, ticker)
        normalized = _enforce_canonical(long)

    # Case 3 — one parquet per ticker (openbb) — multiple files or flat wide multi-file
    else:
        logger.info("Detected format: per-ticker parquet files (Case 3)")
        normalized = _enforce_canonical(_from_per_ticker(raw_parquets))

    return normalized, _make_manifest(normalized)


def _make_manifest(df: pd.DataFrame) -> dict:
    if df.empty:
        return _build_empty_manifest()
    tickers = sorted(df["ticker"].dropna().unique().tolist())
    date_min = df["date"].min()
    date_max = df["date"].max()
    date_range = [
        date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else "",
        date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else "",
    ]
    return {
        "file": "prices.parquet",
        "tickers": tickers,
        "date_range": date_range,
        "row_count": len(df),
        "columns": _CANONICAL_COLS,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(state: AgentState) -> AgentState:
    """
    Normalize all parquet files in state["data_dir"] into canonical
    prices.parquet + data_manifest.json. Pure Python — no LLM calls.
    """
    data_dir_str = state.get("data_dir", "")
    flags: list[str] = []

    if not data_dir_str:
        flags.append("[normalizer] data_dir is empty — skipping normalization")
        manifest = _build_empty_manifest()
        return {**state, "data_manifest": manifest, "flags": flags}

    data_dir = pathlib.Path(data_dir_str)

    try:
        normalized_df, manifest = _normalize_data_dir(data_dir)
    except Exception as exc:
        logger.exception("Normalizer failed with unexpected error")
        flags.append(f"[normalizer] Unexpected error: {exc}")
        manifest = _build_empty_manifest()
        return {**state, "data_manifest": manifest, "flags": flags}

    # Write canonical prices.parquet
    prices_path = data_dir / "prices.parquet"
    if not normalized_df.empty:
        try:
            normalized_df.to_parquet(prices_path, index=False)
            flags.append(
                f"[normalizer] Wrote {len(normalized_df):,} rows × {len(normalized_df.columns)} cols "
                f"to {prices_path.name} — tickers={manifest['tickers']}, "
                f"date_range={manifest['date_range']}"
            )
        except Exception as exc:
            flags.append(f"[normalizer] Failed to write prices.parquet: {exc}")
    else:
        flags.append("[normalizer] Normalized DataFrame is empty — no prices.parquet written")

    # Write data_manifest.json
    manifest_path = data_dir / "data_manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as exc:
        flags.append(f"[normalizer] Failed to write data_manifest.json: {exc}")

    if manifest.get("row_count", 0) == 0:
        flags.append("[normalizer] row_count=0 — data gate will route to reporter")

    return {
        **state,
        "data_manifest": manifest,
        "flags": flags,
    }
