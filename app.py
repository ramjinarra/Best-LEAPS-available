import os
import math
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.stats import norm
import streamlit as st
import yfinance as yf

# ==============================================================================
# PAGE CONFIGURATION & CYBER-QUANT TERMINAL STYLING
# ==============================================================================
st.set_page_config(
    page_title="Memory Supercycle LEAPS & Velocity Terminal (MU | DRAM | RAM | MUU)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .metric-card {
        background: linear-gradient(145deg, #111827, #1e293b);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        margin-bottom: 12px;
    }
    .winner-card-spread {
        background: linear-gradient(145deg, #062e22, #0f172a);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .winner-card-naked {
        background: linear-gradient(145deg, #1e1b4b, #0f172a);
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
    }
    .badge-spread {
        background-color: #059669;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-naked {
        background-color: #4f46e5;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .sub-label {
        color: #94a3b8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .big-stat {
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 4px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# MEMORY SUPERCYCLE WATCHLIST METADATA
# ==============================================================================
MEMORY_TICKERS: Dict[str, Dict] = {
    "MU": {
        "name": "Micron Technology, Inc.",
        "leverage": 1.0,
        "type": "Common Stock (1x)",
        "description": "Primary HBM & DRAM semiconductor leader; deepest LEAPS liquidity out 1–2+ years.",
    },
    "DRAM": {
        "name": "Roundhill Memory ETF",
        "leverage": 1.0,
        "type": "Sector ETF (1x)",
        "description": "Concentrated global memory basket (Micron, SK Hynix, Samsung, NAND/DRAM peers).",
    },
    "RAM": {
        "name": "Roundhill T-REX 2X Long DRAM Daily Target ETF",
        "leverage": 2.0,
        "type": "Leveraged ETF (2x Daily)",
        "description": "200% daily leveraged exposure to the DRAM ETF; subject to daily compounding & volatility drag.",
    },
    "MUU": {
        "name": "Direxion Daily MU Bull 2X Shares",
        "leverage": 2.0,
        "type": "Leveraged ETF (2x Daily)",
        "description": "200% daily leveraged single-stock ETF tracking Micron (MU); high velocity, subject to beta decay.",
    },
}

# ==============================================================================
# BLACK-SCHOLES QUANT & FORWARD VELOCITY RE-PRICING ENGINE
# ==============================================================================
def bs_call_price_and_greeks(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float, float, float, float]:
    """
    Returns (Call Price, Delta, Gamma, Daily Theta, Vega per 1% IV)
    Handles T <= 0 or sigma <= 0 gracefully.
    """
    intrinsic = max(0.0, S - K)
    if T <= 1e-5 or sigma <= 1e-4 or S <= 0 or K <= 0:
        delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        return intrinsic, delta, 0.0, 0.0, 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    price = S * nd1 - K * math.exp(-r * T) * nd2
    delta = nd1
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    theta_annual = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * math.exp(-r * T) * nd2
    theta_daily = theta_annual / 365.0
    vega = (S * pdf_d1 * sqrt_T) / 100.0

    return max(price, intrinsic), delta, gamma, theta_daily, vega


def implied_prob_above(S: float, target_price: float, T: float, r: float, sigma: float) -> float:
    """Risk-neutral probability that underlying >= target_price at horizon T (in years)."""
    if target_price <= 0 or S <= 0:
        return 0.0
    if T <= 1e-5 or sigma <= 1e-4:
        return 100.0 if S >= target_price else 0.0
    sqrt_T = math.sqrt(T)
    d2 = (math.log(S / target_price) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(np.clip(norm.cdf(d2) * 100.0, 0.0, 100.0))


def adjust_target_for_leverage_decay(
    spot: float,
    base_return_pct: float,
    leverage: float,
    annual_vol: float,
    horizon_days: int,
    apply_decay: bool = True,
) -> float:
    """
    Projects target price on horizon_days.
    For 2x daily reset ETFs (RAM, MUU), continuous compounding introduces
    a volatility drag approximately equal to -0.5 * L * (L - 1) * sigma_base^2 * T.
    """
    T_horizon = max(horizon_days, 1) / 365.0
    gross_move = base_return_pct / 100.0

    if leverage <= 1.0 or not apply_decay:
        return spot * (1.0 + gross_move)

    # Approximate underlying 1x volatility from the ETF's implied vol
    base_vol = annual_vol / leverage
    vol_drag = 0.5 * leverage * (leverage - 1.0) * (base_vol**2) * T_horizon
    effective_return = max(-0.95, gross_move - vol_drag)
    return spot * (1.0 + effective_return)


# ==============================================================================
# DUAL DATA ENGINE: ALPACA LIVE OPTIONS API + YFINANCE FALLBACK
# ==============================================================================
def parse_occ_symbol(occ: str) -> Optional[Tuple[str, str, str, float]]:
    """
    Parses OCC option symbol (e.g. MU270115C00150000) into
    (underlying, exp_date_str 'YYYY-MM-DD', opt_type 'C'/'P', strike).
    """
    try:
        occ = occ.strip()
        # Last 8 chars = strike * 1000, preceding 1 char = C/P, preceding 6 chars = YYMMDD
        strike_raw = occ[-8:]
        opt_type = occ[-9]
        date_raw = occ[-15:-9]
        underlying = occ[:-15]
        strike = int(strike_raw) / 1000.0
        exp_dt = dt.datetime.strptime(date_raw, "%y%m%d").strftime("%Y-%m-%d")
        return underlying, exp_dt, opt_type, strike
    except Exception:
        return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_underlying_snapshot(symbol: str) -> Dict:
    """Fetches spot price, 30d historical volatility, and 20d price velocity via yfinance."""
    tkr = yf.Ticker(symbol)
    hist = tkr.history(period="6mo", interval="1d")
    if hist is None or hist.empty:
        raise ValueError(f"Unable to retrieve historical price data for {symbol}.")

    hist = hist.dropna(subset=["Close"])
    spot = float(hist["Close"].iloc[-1])
    prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else spot
    day_chg_pct = ((spot - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

    # Historical Volatility (30-day annualized)
    log_rets = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    hv_30d = float(log_rets.tail(30).std() * np.sqrt(252)) if len(log_rets) >= 5 else 0.55

    # Recent 21-trading-day (~1 month) realized velocity (% per month)
    if len(hist) >= 22:
        p_21d = float(hist["Close"].iloc[-22])
        realized_monthly_vel = ((spot - p_21d) / p_21d) * 100.0
    else:
        p_first = float(hist["Close"].iloc[0])
        days_span = max(1, len(hist))
        realized_monthly_vel = (((spot - p_first) / p_first) * (21.0 / days_span)) * 100.0

    high_52w = float(hist["High"].max())
    low_52w = float(hist["Low"].min())

    return {
        "symbol": symbol,
        "spot": spot,
        "day_chg_pct": day_chg_pct,
        "hv_30d": hv_30d,
        "realized_monthly_vel": realized_monthly_vel,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "history": hist,
    }


@st.cache_data(ttl=180, show_spinner=False)
def fetch_alpaca_options_chain(
    symbol: str,
    api_key: str,
    api_secret: str,
    feed: str = "indicative",
) -> pd.DataFrame:
    """
    Fetches live call option snapshots from Alpaca Options Data API v1beta1.
    Automatically paginates up to 1,000 contracts.
    """
    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"
    headers = {
        "APCA-API-KEY-ID": api_key.strip(),
        "APCA-API-SECRET-KEY": api_secret.strip(),
        "Accept": "application/json",
    }
    today_str = dt.date.today().strftime("%Y-%m-%d")
    params = {
        "feed": feed,
        "type": "call",
        "expiration_date_gte": today_str,
        "limit": 1000,
    }

    rows = []
    page_token = None
    pages_fetched = 0

    while pages_fetched < 5:
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(url, headers=headers, params=params, timeout=12)
        if resp.status_code != 200:
            raise RuntimeError(f"Alpaca API HTTP {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        snapshots = payload.get("snapshots", {})
        for occ_sym, snap in snapshots.items():
            parsed = parse_occ_symbol(occ_sym)
            if not parsed or parsed[2] != "C":
                continue
            _, exp_str, _, strike = parsed

            quote = snap.get("latestQuote") or {}
            trade = snap.get("latestTrade") or {}
            greeks = snap.get("greeks") or {}

            bid = float(quote.get("bp") or 0.0)
            ask = float(quote.get("ap") or 0.0)
            last = float(trade.get("p") or 0.0)
            iv = float(snap.get("impliedVolatility") or 0.0)

            rows.append(
                {
                    "contractSymbol": occ_sym,
                    "expiration": exp_str,
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "lastPrice": last,
                    "impliedVolatility": iv,
                    "volume": int(trade.get("s") or 0),
                    "openInterest": 0,
                    "alpaca_delta": greeks.get("delta"),
                    "alpaca_theta": greeks.get("theta"),
                    "source": f"Alpaca ({feed.upper()})",
                }
            )

        page_token = payload.get("next_page_token")
        pages_fetched += 1
        if not page_token:
            break

    return pd.DataFrame(rows)


@st.cache_data(ttl=180, show_spinner=False)
def fetch_yfinance_options_chain(symbol: str) -> pd.DataFrame:
    """Fetches full Call option chains across all available expirations via Yahoo Finance."""
    tkr = yf.Ticker(symbol)
    expirations = tkr.options
    if not expirations:
        return pd.DataFrame()

    frames = []
    for exp in expirations:
        try:
            chain = tkr.option_chain(exp)
            calls = chain.calls.copy()
            if calls.empty:
                continue
            calls["expiration"] = exp
            calls["source"] = "Yahoo Finance"
            frames.append(calls)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in ["bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def get_enriched_calls_chain(
    symbol: str,
    spot: float,
    hv_30d: float,
    risk_free_rate: float,
    data_provider: str,
    alpaca_key: str,
    alpaca_secret: str,
    alpaca_feed: str,
) -> Tuple[pd.DataFrame, str]:
    """
    Master loader: fetches from Alpaca or Yahoo Finance, cleans stale/after-hours
    quotes, computes DTE, mid-price, and Black-Scholes Greeks.
    """
    used_source = "Yahoo Finance"
    df = pd.DataFrame()

    if data_provider in ("Alpaca Live (Auto-Fallback to Yahoo)", "Alpaca Only") and alpaca_key and alpaca_secret:
        try:
            df = fetch_alpaca_options_chain(symbol, alpaca_key, alpaca_secret, alpaca_feed)
            if not df.empty:
                used_source = f"Alpaca Live ({alpaca_feed.upper()})"
        except Exception as e:
            if data_provider == "Alpaca Only":
                raise e
            st.sidebar.warning(f"Alpaca fallback to Yahoo for {symbol}: {e}")

    if df.empty:
        df = fetch_yfinance_options_chain(symbol)
        used_source = "Yahoo Finance (yfinance)"

    if df.empty:
        return pd.DataFrame(), used_source

    today = dt.date.today()
    df["exp_date"] = pd.to_datetime(df["expiration"]).dt.date
    df["dte"] = df["exp_date"].apply(lambda d: (d - today).days)
    df = df[df["dte"] >= 1].copy()

    if df.empty:
        return pd.DataFrame(), used_source

    # Compute reliable mark / mid price (handling after-hours 0 bid/ask gracefully)
    def calc_mid(row):
        b, a, last = float(row["bid"]), float(row["ask"]), float(row["lastPrice"])
        if b > 0 and a > 0 and a >= b:
            return 0.5 * (b + a)
        if a > 0 and last > 0:
            return 0.5 * (a + last)
        if last > 0:
            return last
        if a > 0:
            return a
        if b > 0:
            return b
        return 0.0

    df["mid"] = df.apply(calc_mid, axis=1)
    # Natural execution prices: buy at slightly above mid (or ask), sell at slightly below mid
    df["buy_fill"] = np.where(
        (df["ask"] > 0) & (df["bid"] > 0) & (df["ask"] >= df["bid"]),
        df["mid"] + 0.15 * (df["ask"] - df["bid"]),
        df["mid"],
    )
    df["sell_fill"] = np.where(
        (df["ask"] > 0) & (df["bid"] > 0) & (df["ask"] >= df["bid"]),
        df["mid"] - 0.15 * (df["ask"] - df["bid"]),
        df["mid"],
    )

    df = df[df["mid"] > 0.03].copy()

    # Clean Implied Volatility (fallback to HV30 if IV is missing or extreme)
    df["iv_clean"] = np.where(
        (df["impliedVolatility"] > 0.05) & (df["impliedVolatility"] < 4.5),
        df["impliedVolatility"],
        max(0.25, hv_30d),
    )

    # Calculate Black-Scholes Greeks & Theoretical Price
    deltas, gammas, thetas, vegas = [], [], [], []
    for _, row in df.iterrows():
        T = row["dte"] / 365.0
        _, d, g, th, v = bs_call_price_and_greeks(
            spot, float(row["strike"]), T, risk_free_rate, float(row["iv_clean"])
        )
        deltas.append(d)
        gammas.append(g)
        thetas.append(th)
        vegas.append(v)

    df["delta"] = deltas
    df["gamma"] = gammas
    df["theta"] = thetas
    df["vega"] = vegas
    df["bid_ask_spread_pct"] = np.where(
        (df["ask"] > 0) & (df["bid"] >= 0) & (df["mid"] > 0),
        ((df["ask"] - df["bid"]) / df["mid"]) * 100.0,
        0.0,
    )

    return df.sort_values(["dte", "strike"]).reset_index(drop=True), used_source


# ==============================================================================
# STRATEGY ENGINE: LEAPS NAKED CALLS VS. BULL CALL DEBIT SPREADS
# ==============================================================================
def build_leaps_and_spreads_candidates(
    df_calls: pd.DataFrame,
    symbol: str,
    spot: float,
    leverage: float,
    target_price_horizon: float,
    target_price_exp_rate_monthly: float,
    horizon_days: int,
    dte_min: int,
    dte_max: int,
    min_roi_target: float,
    max_contract_cost: float,
    risk_free_rate: float,
    apply_leverage_decay: bool,
    use_natural_fills: bool,
) -> Tuple[pd.DataFrame, bool]:
    """
    Evaluates both Naked Long Calls and Bull Call Debit Spreads across available expirations.
    Returns (candidates_df, used_adaptive_dte_fallback).
    """
    if df_calls.empty:
        return pd.DataFrame(), False

    # Filter DTE window; if empty (e.g. newer ETFs like DRAM/RAM/MUU), adaptively pick top 4 longest expirations
    sub = df_calls[(df_calls["dte"] >= dte_min) & (df_calls["dte"] <= dte_max)].copy()
    used_fallback = False
    if sub.empty:
        used_fallback = True
        longest_exps = sorted(df_calls["dte"].unique(), reverse=True)[:4]
        sub = df_calls[df_calls["dte"].isin(longest_exps)].copy()

    candidates = []

    for exp_str, grp in sub.groupby("expiration"):
        grp = grp.sort_values("strike").reset_index(drop=True)
        dte = int(grp["dte"].iloc[0])
        T_exp = dte / 365.0

        # Effective holding period cannot exceed expiration DTE
        eff_horizon_days = min(horizon_days, dte)
        T_rem_at_horizon = max(0.0, (dte - eff_horizon_days) / 365.0)

        # Project stock price at expiration if velocity continues, or cap at a realistic multiple
        months_to_exp = dte / 30.4375
        proj_move_exp_pct = target_price_exp_rate_monthly * months_to_exp
        avg_iv = float(grp["iv_clean"].median())
        target_price_at_exp = adjust_target_for_leverage_decay(
            spot=spot,
            base_return_pct=proj_move_exp_pct,
            leverage=leverage,
            annual_vol=avg_iv,
            horizon_days=dte,
            apply_decay=apply_leverage_decay,
        )

        # Restrict Long leg strikes to sensible range (0.55x spot to 1.45x spot)
        long_pool = grp[(grp["strike"] >= spot * 0.55) & (grp["strike"] <= spot * 1.45)].copy()

        # ----------------------------------------------------------------------
        # 1. EVALUATE NAKED LONG CALLS (LEAPS)
        # ----------------------------------------------------------------------
        for _, l_row in long_pool.iterrows():
            k_long = float(l_row["strike"])
            entry_debit = float(l_row["buy_fill"] if use_natural_fills else l_row["mid"])
            if entry_debit <= 0.05 or (entry_debit * 100.0) > max_contract_cost:
                continue

            iv_long = float(l_row["iv_clean"])

            # Value of Naked Call at Target Horizon Date (Black-Scholes Re-Pricing)
            val_horizon, _, _, _, _ = bs_call_price_and_greeks(
                target_price_horizon, k_long, T_rem_at_horizon, risk_free_rate, iv_long
            )
            roi_horizon = ((val_horizon - entry_debit) / entry_debit) * 100.0
            profit_horizon_dlr = (val_horizon - entry_debit) * 100.0

            # Value of Naked Call at Expiration if Target Horizon Price is held
            val_exp_at_target = max(0.0, target_price_horizon - k_long)
            roi_exp_at_target = ((val_exp_at_target - entry_debit) / entry_debit) * 100.0

            # Value at Expiration under continuous monthly velocity projection
            val_exp_vel = max(0.0, target_price_at_exp - k_long)
            roi_exp_vel = ((val_exp_vel - entry_debit) / entry_debit) * 100.0

            breakeven = k_long + entry_debit
            req_move_be_pct = ((breakeven - spot) / spot) * 100.0
            req_vel_monthly_be = req_move_be_pct / max(0.25, months_to_exp)

            pop_horizon = implied_prob_above(
                spot, k_long + entry_debit * 0.35, eff_horizon_days / 365.0, risk_free_rate, iv_long
            )
            pop_exp = implied_prob_above(spot, breakeven, T_exp, risk_free_rate, iv_long)

            roi_velocity_per_day = roi_horizon / max(1, eff_horizon_days)
            # Composite Velocity Edge Score balances Horizon ROI, PoP, and liquidity
            spread_penalty = min(25.0, float(l_row["bid_ask_spread_pct"]) * 0.35)
            velocity_edge_score = (
                (roi_horizon * 0.55)
                + (roi_velocity_per_day * 8.0)
                + (pop_horizon * 0.45)
                - spread_penalty
            )

            candidates.append(
                {
                    "Ticker": symbol,
                    "Strategy": "Naked Call",
                    "Structure": f"{exp_str} ${k_long:g}C",
                    "Expiration": exp_str,
                    "DTE": dte,
                    "Long Strike": k_long,
                    "Short Strike": np.nan,
                    "Width ($)": np.nan,
                    "Net Debit ($)": round(entry_debit, 2),
                    "Contract Cost ($)": round(entry_debit * 100.0, 0),
                    "Max Value ($)": np.inf,
                    "Horizon Value ($)": round(val_horizon, 2),
                    "Horizon Profit ($)": round(profit_horizon_dlr, 0),
                    "Horizon ROI (%)": round(roi_horizon, 1),
                    "ROI at Target (Exp %)": round(roi_exp_at_target, 1),
                    "Projected Exp ROI (%)": round(roi_exp_vel, 1),
                    "ROI Velocity (%/Day)": round(roi_velocity_per_day, 2),
                    "Breakeven ($)": round(breakeven, 2),
                    "Req Move to BE (%)": round(req_move_be_pct, 1),
                    "Req Monthly Vel (%/mo)": round(req_vel_monthly_be, 2),
                    "Net Delta": round(float(l_row["delta"]), 2),
                    "Net Theta ($/d)": round(float(l_row["theta"]) * 100.0, 2),
                    "IV (%)": round(iv_long * 100.0, 1),
                    "PoP Horizon (%)": round(pop_horizon, 1),
                    "PoP Exp (%)": round(pop_exp, 1),
                    "Bid-Ask Spread (%)": round(float(l_row["bid_ask_spread_pct"]), 1),
                    "Velocity Score": round(velocity_edge_score, 1),
                    "_iv_short": 0.0,
                }
            )

        # ----------------------------------------------------------------------
        # 2. EVALUATE BULL CALL DEBIT SPREADS (LEAPS VERTICALS)
        # ----------------------------------------------------------------------
        short_pool = grp[
            (grp["strike"] > spot * 0.90) & (grp["strike"] <= max(target_price_horizon * 1.45, spot * 1.85))
        ].copy()

        # Focus long legs on highest-efficiency delta range (0.35 to 0.85)
        spread_long_pool = long_pool[(long_pool["delta"] >= 0.30) & (long_pool["delta"] <= 0.88)]

        for _, l_row in spread_long_pool.iterrows():
            k_long = float(l_row["strike"])
            buy_leg = float(l_row["buy_fill"] if use_natural_fills else l_row["mid"])
            iv_long = float(l_row["iv_clean"])

            valid_shorts = short_pool[short_pool["strike"] > k_long * 1.03]
            # Sample up to 8 well-spaced short strikes per long strike to keep scanner fast
            if len(valid_shorts) > 8:
                idx = np.linspace(0, len(valid_shorts) - 1, 8).astype(int)
                valid_shorts = valid_shorts.iloc[idx]

            for _, s_row in valid_shorts.iterrows():
                k_short = float(s_row["strike"])
                sell_leg = float(s_row["sell_fill"] if use_natural_fills else s_row["mid"])
                iv_short = float(s_row["iv_clean"])

                width = k_short - k_long
                net_debit = buy_leg - sell_leg

                # Filter invalid or poorly priced spreads
                if net_debit <= 0.10 or net_debit >= width * 0.92:
                    continue
                if (net_debit * 100.0) > max_contract_cost:
                    continue

                max_roi_cap = ((width - net_debit) / net_debit) * 100.0
                if max_roi_cap < min_roi_target:
                    continue

                # Re-price both legs at Target Horizon Date
                val_long_h, _, _, _, _ = bs_call_price_and_greeks(
                    target_price_horizon, k_long, T_rem_at_horizon, risk_free_rate, iv_long
                )
                val_short_h, _, _, _, _ = bs_call_price_and_greeks(
                    target_price_horizon, k_short, T_rem_at_horizon, risk_free_rate, iv_short
                )
                spread_val_h = float(np.clip(val_long_h - val_short_h, 0.0, width))

                roi_horizon = ((spread_val_h - net_debit) / net_debit) * 100.0
                profit_horizon_dlr = (spread_val_h - net_debit) * 100.0

                # Intrinsic value at Expiration if stock finishes at Target Horizon Price
                intrinsic_at_target = float(np.clip(target_price_horizon - k_long, 0.0, width))
                roi_exp_at_target = ((intrinsic_at_target - net_debit) / net_debit) * 100.0

                # Intrinsic value at Expiration under continuous velocity
                intrinsic_vel_exp = float(np.clip(target_price_at_exp - k_long, 0.0, width))
                roi_exp_vel = ((intrinsic_vel_exp - net_debit) / net_debit) * 100.0

                breakeven = k_long + net_debit
                req_move_be_pct = ((breakeven - spot) / spot) * 100.0
                req_vel_monthly_be = req_move_be_pct / max(0.25, months_to_exp)

                pop_horizon = implied_prob_above(
                    spot, k_long + net_debit * 0.30, eff_horizon_days / 365.0, risk_free_rate, iv_long
                )
                pop_exp = implied_prob_above(spot, breakeven, T_exp, risk_free_rate, iv_long)

                net_delta = float(l_row["delta"]) - float(s_row["delta"])
                net_theta = (float(l_row["theta"]) - float(s_row["theta"])) * 100.0
                avg_ba_spread = 0.5 * (
                    float(l_row["bid_ask_spread_pct"]) + float(s_row["bid_ask_spread_pct"])
                )

                roi_velocity_per_day = roi_horizon / max(1, eff_horizon_days)
                spread_penalty = min(25.0, avg_ba_spread * 0.35)
                # Spreads get a capital-preservation & lower breakeven bonus in the composite score
                velocity_edge_score = (
                    (roi_horizon * 0.55)
                    + (roi_velocity_per_day * 8.0)
                    + (pop_horizon * 0.50)
                    - spread_penalty
                )

                candidates.append(
                    {
                        "Ticker": symbol,
                        "Strategy": "Bull Call Spread",
                        "Structure": f"{exp_str} ${k_long:g}/${k_short:g}C",
                        "Expiration": exp_str,
                        "DTE": dte,
                        "Long Strike": k_long,
                        "Short Strike": k_short,
                        "Width ($)": round(width, 2),
                        "Net Debit ($)": round(net_debit, 2),
                        "Contract Cost ($)": round(net_debit * 100.0, 0),
                        "Max Value ($)": round(width, 2),
                        "Horizon Value ($)": round(spread_val_h, 2),
                        "Horizon Profit ($)": round(profit_horizon_dlr, 0),
                        "Horizon ROI (%)": round(roi_horizon, 1),
                        "ROI at Target (Exp %)": round(roi_exp_at_target, 1),
                        "Projected Exp ROI (%)": round(roi_exp_vel, 1),
                        "ROI Velocity (%/Day)": round(roi_velocity_per_day, 2),
                        "Breakeven ($)": round(breakeven, 2),
                        "Req Move to BE (%)": round(req_move_be_pct, 1),
                        "Req Monthly Vel (%/mo)": round(req_vel_monthly_be, 2),
                        "Net Delta": round(net_delta, 2),
                        "Net Theta ($/d)": round(net_theta, 2),
                        "IV (%)": round(iv_long * 100.0, 1),
                        "PoP Horizon (%)": round(pop_horizon, 1),
                        "PoP Exp (%)": round(pop_exp, 1),
                        "Bid-Ask Spread (%)": round(avg_ba_spread, 1),
                        "Velocity Score": round(velocity_edge_score, 1),
                        "_iv_short": iv_short,
                    }
                )

    df_out = pd.DataFrame(candidates)
    if df_out.empty:
        return df_out, used_fallback

    # Filter by user's minimum ROI threshold (checking either Horizon ROI or Expiration ROI at target)
    df_filtered = df_out[
        (df_out["Horizon ROI (%)"] >= min_roi_target)
        | (df_out["ROI at Target (Exp %)"] >= min_roi_target)
    ].copy()

    if df_filtered.empty:
        df_filtered = df_out.copy()

    df_filtered = df_filtered.sort_values(
        by=["Velocity Score", "Horizon ROI (%)"], ascending=[False, False]
    ).reset_index(drop=True)

    return df_filtered, used_fallback


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown("## ⚡ LEAPS & Velocity Controls")

    selected_ticker = st.selectbox(
        "Primary Memory Ticker",
        options=list(MEMORY_TICKERS.keys()),
        index=0,
        format_func=lambda x: f"{x} — {MEMORY_TICKERS[x]['type']}",
    )

    custom_ticker = st.text_input("Or Enter Custom Ticker (Optional)", value="").strip().upper()
    active_symbol = custom_ticker if custom_ticker else selected_ticker

    st.markdown("---")
    st.markdown("### 📡 Live Options Data Feed")
    data_provider = st.radio(
        "Options Chain Engine",
        options=[
            "Yahoo Finance (Zero-Config Default)",
            "Alpaca Live (Auto-Fallback to Yahoo)",
            "Alpaca Only",
        ],
        index=0,
        help="Yahoo Finance works out-of-the-box. Select Alpaca Live and input your API keys for real-time OPRA/Indicative snapshots.",
    )

    default_apca_key = os.environ.get("APCA_API_KEY_ID", "")
    default_apca_secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if hasattr(st, "secrets"):
        try:
            default_apca_key = st.secrets.get("APCA_API_KEY_ID", default_apca_key)
            default_apca_secret = st.secrets.get("APCA_API_SECRET_KEY", default_apca_secret)
        except Exception:
            pass

    alpaca_key, alpaca_secret, alpaca_feed = "", "", "indicative"
    if "Alpaca" in data_provider:
        alpaca_key = st.text_input("Alpaca API Key ID", value=default_apca_key, type="password")
        alpaca_secret = st.text_input("Alpaca Secret Key", value=default_apca_secret, type="password")
        alpaca_feed = st.selectbox(
            "Alpaca Options Feed",
            options=["indicative", "opra"],
            index=0,
            help="Use 'opra' if your Alpaca account has OPRA real-time subscription enabled, otherwise 'indicative' (free tier).",
        )

    st.markdown("---")
    st.markdown("### 🎯 Date & Velocity Parameters")

    dte_range = st.slider(
        "Option Expiration Window (DTE)",
        min_value=14,
        max_value=900,
        value=(150, 750),
        step=15,
        help="Standard LEAPS are 180–750+ DTE. If a newer ETF (DRAM, RAM, MUU) only has shorter-dated expirations, the terminal auto-adapts.",
    )

    horizon_days = st.slider(
        "Target Evaluation Horizon (Days Held)",
        min_value=7,
        max_value=365,
        value=60,
        step=5,
        help="Re-prices LEAPS Calls & Spreads on this future target date using Black-Scholes so you see early mark-to-market ROI before expiration.",
    )

    velocity_mode = st.radio(
        "Target Move Input Mode",
        options=["Monthly Velocity (%/Month)", "Direct Target Move (%) by Horizon"],
        index=0,
    )

    if velocity_mode == "Monthly Velocity (%/Month)":
        monthly_vel_pct = st.slider(
            "Expected Underlying 1x Velocity (%/Month)",
            min_value=2.0,
            max_value=35.0,
            value=10.0,
            step=0.5,
            help="Expected monthly % appreciation for 1x underlying (MU / DRAM). Automatically scaled by 2x for RAM / MUU.",
        )
        base_horizon_move_pct = monthly_vel_pct * (horizon_days / 30.4375)
    else:
        base_horizon_move_pct = st.slider(
            f"Expected 1x Move (%) in {horizon_days} Days",
            min_value=5.0,
            max_value=150.0,
            value=22.0,
            step=1.0,
        )
        monthly_vel_pct = base_horizon_move_pct / (horizon_days / 30.4375)

    st.markdown("---")
    st.markdown("### 🛡️ Risk & Execution Filters")
    min_roi_target = st.slider("Minimum Target ROI (%)", min_value=20, max_value=300, value=40, step=10)
    max_contract_cost = st.number_input(
        "Max Capital Per 1 Contract / Spread ($)",
        min_value=100,
        max_value=25000,
        value=3500,
        step=250,
    )
    apply_leverage_decay = st.checkbox(
        "Model 2x Daily ETF Volatility Decay (RAM / MUU)",
        value=True,
        help="Adjusts projected target prices on 2x daily ETFs (RAM, MUU) for continuous compounding variance drag.",
    )
    use_natural_fills = st.checkbox(
        "Conservative Slippage Fills (Buy > Mid, Sell < Mid)",
        value=True,
    )
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.25, step=0.25) / 100.0


# ==============================================================================
# LOAD UNDERLYING & OPTION CHAIN FOR ACTIVE SYMBOL
# ==============================================================================
tkr_meta = MEMORY_TICKERS.get(
    active_symbol,
    {
        "name": f"{active_symbol} Custom Equity/ETF",
        "leverage": 1.0,
        "type": "Custom Ticker (1x Assumed)",
        "description": "User-specified ticker analyzed with the LEAPS & Velocity Quant Engine.",
    },
)
leverage_mult = float(tkr_meta["leverage"])

with st.spinner(f"Fetching live market & option chains for {active_symbol}..."):
    try:
        snap = fetch_underlying_snapshot(active_symbol)
        df_calls, active_source = get_enriched_calls_chain(
            symbol=active_symbol,
            spot=snap["spot"],
            hv_30d=snap["hv_30d"],
            risk_free_rate=risk_free_rate,
            data_provider=data_provider,
            alpaca_key=alpaca_key,
            alpaca_secret=alpaca_secret,
            alpaca_feed=alpaca_feed,
        )
    except Exception as err:
        st.error(f"Error loading data for `{active_symbol}`: {err}")
        st.stop()

spot_price = snap["spot"]
scaled_horizon_move_pct = base_horizon_move_pct * leverage_mult
scaled_monthly_vel_pct = monthly_vel_pct * leverage_mult

target_price_horizon = adjust_target_for_leverage_decay(
    spot=spot_price,
    base_return_pct=scaled_horizon_move_pct,
    leverage=leverage_mult,
    annual_vol=snap["hv_30d"],
    horizon_days=horizon_days,
    apply_decay=apply_leverage_decay,
)
effective_horizon_move_pct = ((target_price_horizon - spot_price) / spot_price) * 100.0
target_date_str = (dt.date.today() + dt.timedelta(days=horizon_days)).strftime("%b %d, %Y")

# ==============================================================================
# HEADER & TOP METRIC BANNER
# ==============================================================================
st.markdown(
    f"## ⚡ {active_symbol} LEAPS Bull Call Spread vs. Naked Call Velocity Terminal"
)
st.caption(
    f"**{tkr_meta['name']}** ({tkr_meta['type']}) • {tkr_meta['description']} • **Live Feed:** `{active_source}`"
)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    chg_color = "#10b981" if snap["day_chg_pct"] >= 0 else "#ef4444"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="sub-label">{active_symbol} Spot Price</div>
            <div class="big-stat">${spot_price:,.2f}</div>
            <div style="color:{chg_color}; font-size:0.85rem; font-weight:600;">
                {snap['day_chg_pct']:+.2f}% Today | 52W: ${snap['low_52w']:,.1f}–${snap['high_52w']:,.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="sub-label">Target @ {horizon_days}d ({target_date_str})</div>
            <div class="big-stat">${target_price_horizon:,.2f}</div>
            <div style="color:#38bdf8; font-size:0.85rem; font-weight:600;">
                +{effective_horizon_move_pct:.1f}% ({leverage_mult:g}x Scaled Move)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="sub-label">21D Realized vs Target Velocity</div>
            <div class="big-stat">{snap['realized_monthly_vel']:+.1f}%/mo</div>
            <div style="color:#a78bfa; font-size:0.85rem; font-weight:600;">
                Model Target: +{scaled_monthly_vel_pct:.1f}%/mo
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m4:
    median_iv = float(df_calls["iv_clean"].median() * 100.0) if not df_calls.empty else 0.0
    hv_pct = snap["hv_30d"] * 100.0
    iv_hv_ratio = median_iv / hv_pct if hv_pct > 0 else 1.0
    iv_tag = "Cheap IV (Long Call Edge)" if iv_hv_ratio < 0.98 else ("Rich IV (Spread Edge)" if iv_hv_ratio > 1.12 else "Fair IV Regime")
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="sub-label">Chain IV vs 30D HV</div>
            <div class="big-stat">{median_iv:.1f}% IV</div>
            <div style="color:#fbbf24; font-size:0.85rem; font-weight:600;">
                30D HV: {hv_pct:.1f}% • {iv_tag}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with m5:
    max_dte_avail = int(df_calls["dte"].max()) if not df_calls.empty else 0
    num_exps = df_calls["expiration"].nunique() if not df_calls.empty else 0
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="sub-label">Options Horizon Coverage</div>
            <div class="big-stat">{max_dte_avail} Max DTE</div>
            <div style="color:#94a3b8; font-size:0.85rem; font-weight:600;">
                {num_exps} Expirations | {len(df_calls):,} Calls Loaded
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if df_calls.empty:
    st.warning(
        f"No active option contracts were returned for `{active_symbol}`. Try switching tickers or checking market feed connectivity."
    )
    st.stop()

# Run the LEAPS & Spreads optimization engine
candidates_df, used_fallback = build_leaps_and_spreads_candidates(
    df_calls=df_calls,
    symbol=active_symbol,
    spot=spot_price,
    leverage=leverage_mult,
    target_price_horizon=target_price_horizon,
    target_price_exp_rate_monthly=scaled_monthly_vel_pct,
    horizon_days=horizon_days,
    dte_min=dte_range[0],
    dte_max=dte_range[1],
    min_roi_target=min_roi_target,
    max_contract_cost=max_contract_cost,
    risk_free_rate=risk_free_rate,
    apply_leverage_decay=apply_leverage_decay,
    use_natural_fills=use_natural_fills,
)

if used_fallback:
    st.info(
        f"ℹ️ **Adaptive Horizon Active:** `{active_symbol}` currently has its longest-dated option expirations below {dte_range[0]} DTE (Max available: **{max_dte_avail} DTE**). Automatically evaluating the top furthest-dated expirations on the chain."
    )

if candidates_df.empty:
    st.warning(
        "No contracts matched your exact Cost / DTE filters. Try increasing 'Max Capital Per Contract' or widening your DTE slider in the sidebar."
    )
    st.stop()

# ==============================================================================
# HEAD-TO-HEAD WINNER SPOTLIGHT: BEST SPREAD VS. BEST NAKED CALL
# ==============================================================================
spreads_only = candidates_df[candidates_df["Strategy"] == "Bull Call Spread"].reset_index(drop=True)
naked_only = candidates_df[candidates_df["Strategy"] == "Naked Call"].reset_index(drop=True)

best_overall = candidates_df.iloc[0]
best_spread = spreads_only.iloc[0] if not spreads_only.empty else None
best_naked = naked_only.iloc[0] if not naked_only.empty else None

st.markdown("### 🏆 Head-to-Head Velocity Verdict: Bull Call Spread vs. Naked LEAPS Call")

col_w1, col_w2 = st.columns(2)

with col_w1:
    if best_spread is not None:
        is_top = best_overall["Strategy"] == "Bull Call Spread"
        crown = "👑 OVERALL WINNER — " if is_top else ""
        st.markdown(
            f"""
            <div class="winner-card-spread">
                <span class="badge-spread">{crown}BEST LEAPS BULL CALL SPREAD</span>
                <h3 style="margin:10px 0 4px 0; color:#f8fafc;">{active_symbol} {best_spread['Structure']} ({best_spread['DTE']} DTE)</h3>
                <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:10px;">
                    Net Debit: <b>${best_spread['Net Debit ($)']:.2f}</b> (${best_spread['Contract Cost ($)']:,.0f}/contract) •
                    Width: <b>${best_spread['Width ($)']:.2f}</b> • Breakeven: <b>${best_spread['Breakeven ($)']:.2f}</b> ({best_spread['Req Move to BE (%)']:+.1f}%)
                </p>
                <div style="display:flex; justify-content:space-between; border-top:1px solid #1e293b; padding-top:10px;">
                    <div><span class="sub-label">{horizon_days}d Horizon ROI</span><br><b style="color:#10b981; font-size:1.3rem;">{best_spread['Horizon ROI (%)']:+.1f}%</b> (${best_spread['Horizon Profit ($)']:+,.0f})</div>
                    <div><span class="sub-label">ROI @ Exp (Target)</span><br><b style="color:#34d399; font-size:1.3rem;">{best_spread['ROI at Target (Exp %)']:+.1f}%</b></div>
                    <div><span class="sub-label">ROI Velocity</span><br><b style="color:#f8fafc; font-size:1.3rem;">{best_spread['ROI Velocity (%/Day)']:+.2f}%/d</b></div>
                    <div><span class="sub-label">Net Δ / Θ</span><br><b style="color:#94a3b8; font-size:1.1rem;">{best_spread['Net Delta']:.2f}Δ / ${best_spread['Net Theta ($/d)']:.2f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No Bull Call Spreads matched current filter criteria.")

with col_w2:
    if best_naked is not None:
        is_top = best_overall["Strategy"] == "Naked Call"
        crown = "👑 OVERALL WINNER — " if is_top else ""
        st.markdown(
            f"""
            <div class="winner-card-naked">
                <span class="badge-naked">{crown}BEST NAKED LEAPS CALL</span>
                <h3 style="margin:10px 0 4px 0; color:#f8fafc;">{active_symbol} {best_naked['Structure']} ({best_naked['DTE']} DTE)</h3>
                <p style="color:#cbd5e1; font-size:0.9rem; margin-bottom:10px;">
                    Call Premium: <b>${best_naked['Net Debit ($)']:.2f}</b> (${best_naked['Contract Cost ($)']:,.0f}/contract) •
                    Uncapped Upside • Breakeven: <b>${best_naked['Breakeven ($)']:.2f}</b> ({best_naked['Req Move to BE (%)']:+.1f}%)
                </p>
                <div style="display:flex; justify-content:space-between; border-top:1px solid #1e293b; padding-top:10px;">
                    <div><span class="sub-label">{horizon_days}d Horizon ROI</span><br><b style="color:#818cf8; font-size:1.3rem;">{best_naked['Horizon ROI (%)']:+.1f}%</b> (${best_naked['Horizon Profit ($)']:+,.0f})</div>
                    <div><span class="sub-label">ROI @ Exp (Target)</span><br><b style="color:#a5b4fc; font-size:1.3rem;">{best_naked['ROI at Target (Exp %)']:+.1f}%</b></div>
                    <div><span class="sub-label">ROI Velocity</span><br><b style="color:#f8fafc; font-size:1.3rem;">{best_naked['ROI Velocity (%/Day)']:+.2f}%/d</b></div>
                    <div><span class="sub-label">Call Δ / Θ</span><br><b style="color:#94a3b8; font-size:1.1rem;">{best_naked['Net Delta']:.2f}Δ / ${best_naked['Net Theta ($/d)']:.2f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No Naked Calls matched current filter criteria.")

# ==============================================================================
# INTERACTIVE TERMINAL TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        f"📋 {active_symbol} Ranked Strategy Leaderboard",
        "📈 Velocity Trajectory & Date × Price Heatmap",
        "🌐 Multi-Ticker Memory Scan (MU | DRAM | RAM | MUU)",
        "🧮 Custom LEAPS Spread vs. Call Lab",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: STRATEGY LEADERBOARD
# ------------------------------------------------------------------------------
with tab1:
    f_col1, f_col2, f_col3 = st.columns([1.5, 1.5, 2])
    with f_col1:
        strat_filter = st.selectbox(
            "Filter Strategy Type",
            options=["All (Head-to-Head)", "Bull Call Spread Only", "Naked Call Only"],
            index=0,
        )
    with f_col2:
        sort_metric = st.selectbox(
            "Rank Leaderboard By",
            options=[
                "Velocity Score",
                "Horizon ROI (%)",
                "ROI Velocity (%/Day)",
                "ROI at Target (Exp %)",
                "Projected Exp ROI (%)",
                "PoP Horizon (%)",
            ],
            index=0,
        )
    with f_col3:
        exp_options = ["All Expirations"] + sorted(candidates_df["Expiration"].unique().tolist())
        exp_filter = st.selectbox("Filter Expiration Date", options=exp_options, index=0)

    view_df = candidates_df.copy()
    if strat_filter == "Bull Call Spread Only":
        view_df = view_df[view_df["Strategy"] == "Bull Call Spread"]
    elif strat_filter == "Naked Call Only":
        view_df = view_df[view_df["Strategy"] == "Naked Call"]

    if exp_filter != "All Expirations":
        view_df = view_df[view_df["Expiration"] == exp_filter]

    view_df = view_df.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)

    display_cols = [
        "Strategy",
        "Structure",
        "Expiration",
        "DTE",
        "Net Debit ($)",
        "Contract Cost ($)",
        "Horizon Value ($)",
        "Horizon Profit ($)",
        "Horizon ROI (%)",
        "ROI at Target (Exp %)",
        "ROI Velocity (%/Day)",
        "Breakeven ($)",
        "Req Move to BE (%)",
        "Net Delta",
        "Net Theta ($/d)",
        "IV (%)",
        "PoP Horizon (%)",
        "Bid-Ask Spread (%)",
        "Velocity Score",
    ]

    st.dataframe(
        view_df[display_cols].head(100),
        use_container_width=True,
        height=480,
    )

    csv_bytes = view_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download {active_symbol} LEAPS Velocity Leaderboard (CSV)",
        data=csv_bytes,
        file_name=f"{active_symbol}_LEAPS_Velocity_{dt.date.today()}.csv",
        mime="text/csv",
    )

# ------------------------------------------------------------------------------
# TAB 2: PAYOFF CURVES & DATE x PRICE HEATMAP
# ------------------------------------------------------------------------------
with tab2:
    if best_spread is not None and best_naked is not None:
        st.markdown(
            f"#### 📊 Direct ROI Comparison: `{best_spread['Structure']}` (Spread) vs. `{best_naked['Structure']}` (Naked Call)"
        )

        price_grid = np.linspace(spot_price * 0.65, max(target_price_horizon * 1.45, spot_price * 1.65), 80)

        # Compute ROI curves at Horizon Date (Black-Scholes) and at Expiration
        T_rem_sp = max(0.0, (int(best_spread["DTE"]) - horizon_days) / 365.0)
        T_rem_nk = max(0.0, (int(best_naked["DTE"]) - horizon_days) / 365.0)

        sp_roi_h, sp_roi_exp = [], []
        nk_roi_h, nk_roi_exp = [], []

        k_l_sp = float(best_spread["Long Strike"])
        k_s_sp = float(best_spread["Short Strike"])
        deb_sp = float(best_spread["Net Debit ($)"])
        iv_l_sp = float(best_spread["IV (%)"]) / 100.0
        iv_s_sp = float(best_spread["_iv_short"])

        k_l_nk = float(best_naked["Long Strike"])
        deb_nk = float(best_naked["Net Debit ($)"])
        iv_l_nk = float(best_naked["IV (%)"]) / 100.0

        for p in price_grid:
            # Spread at Horizon
            v1, _, _, _, _ = bs_call_price_and_greeks(p, k_l_sp, T_rem_sp, risk_free_rate, iv_l_sp)
            v2, _, _, _, _ = bs_call_price_and_greeks(p, k_s_sp, T_rem_sp, risk_free_rate, iv_s_sp)
            val_sp_h = np.clip(v1 - v2, 0.0, k_s_sp - k_l_sp)
            sp_roi_h.append(((val_sp_h - deb_sp) / deb_sp) * 100.0)

            # Spread at Expiration
            val_sp_e = np.clip(p - k_l_sp, 0.0, k_s_sp - k_l_sp)
            sp_roi_exp.append(((val_sp_e - deb_sp) / deb_sp) * 100.0)

            # Naked Call at Horizon
            vn, _, _, _, _ = bs_call_price_and_greeks(p, k_l_nk, T_rem_nk, risk_free_rate, iv_l_nk)
            nk_roi_h.append(((vn - deb_nk) / deb_nk) * 100.0)

            # Naked Call at Expiration
            vn_e = max(0.0, p - k_l_nk)
            nk_roi_exp.append(((vn_e - deb_nk) / deb_nk) * 100.0)

        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Scatter(
                x=price_grid,
                y=sp_roi_h,
                name=f"Bull Call Spread ({horizon_days}d Horizon)",
                line=dict(color="#10b981", width=3),
            )
        )
        fig_comp.add_trace(
            go.Scatter(
                x=price_grid,
                y=sp_roi_exp,
                name="Bull Call Spread (At Expiration)",
                line=dict(color="#10b981", width=1.5, dash="dot"),
            )
        )
        fig_comp.add_trace(
            go.Scatter(
                x=price_grid,
                y=nk_roi_h,
                name=f"Naked Call ({horizon_days}d Horizon)",
                line=dict(color="#6366f1", width=3),
            )
        )
        fig_comp.add_trace(
            go.Scatter(
                x=price_grid,
                y=nk_roi_exp,
                name="Naked Call (At Expiration)",
                line=dict(color="#6366f1", width=1.5, dash="dot"),
            )
        )

        fig_comp.add_vline(
            x=spot_price,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text=f"Spot ${spot_price:.2f}",
        )
        fig_comp.add_vline(
            x=target_price_horizon,
            line_dash="dash",
            line_color="#38bdf8",
            annotation_text=f"Target ${target_price_horizon:.2f}",
        )
        fig_comp.add_hline(y=0, line_color="#475569", line_width=1)

        fig_comp.update_layout(
            template="plotly_dark",
            height=430,
            xaxis_title=f"{active_symbol} Underlying Price ($)",
            yaxis_title="Return on Investment (ROI %)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # 2D Date x Price Heatmap for the #1 Overall Winner
        st.markdown(
            f"#### 🔥 2D Holding Period × Price ROI (%) Surface — Winner: `{best_overall['Strategy']} ({best_overall['Structure']})`"
        )
        max_h = int(best_overall["DTE"])
        day_steps = np.linspace(1, max_h, 12).astype(int)
        p_steps = np.linspace(spot_price * 0.75, max(target_price_horizon * 1.35, spot_price * 1.50), 15)

        z_matrix = []
        for p_val in p_steps:
            row_z = []
            for d_elapsed in day_steps:
                t_left = max(0.0, (max_h - d_elapsed) / 365.0)
                kl = float(best_overall["Long Strike"])
                deb = float(best_overall["Net Debit ($)"])
                ivl = float(best_overall["IV (%)"]) / 100.0
                v_l, _, _, _, _ = bs_call_price_and_greeks(p_val, kl, t_left, risk_free_rate, ivl)

                if best_overall["Strategy"] == "Bull Call Spread":
                    ks = float(best_overall["Short Strike"])
                    ivs = float(best_overall["_iv_short"])
                    v_s, _, _, _, _ = bs_call_price_and_greeks(p_val, ks, t_left, risk_free_rate, ivs)
                    pos_val = np.clip(v_l - v_s, 0.0, ks - kl)
                else:
                    pos_val = v_l

                roi_cell = ((pos_val - deb) / deb) * 100.0
                row_z.append(round(roi_cell, 1))
            z_matrix.append(row_z)

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                x=[f"Day {d} ({max_h - d} DTE left)" for d in day_steps],
                y=[f"${p:.2f} ({((p - spot_price)/spot_price)*100:+.0f}%)" for p in p_steps],
                colorscale="RdYlGn",
                zmid=0,
                text=z_matrix,
                texttemplate="%{text}%",
                colorbar=dict(title="ROI %"),
            )
        )
        fig_hm.update_layout(
            template="plotly_dark",
            height=470,
            xaxis_title="Days Held / Remaining DTE",
            yaxis_title=f"{active_symbol} Price Level",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: MULTI-TICKER MEMORY SUPERCYCLE MATRIX (MU, DRAM, RAM, MUU)
# ------------------------------------------------------------------------------
with tab3:
    st.markdown(
        "#### 🌐 Cross-Ticker Memory Supercycle Scanner (`MU` vs `DRAM` vs `RAM` vs `MUU`)"
    )
    st.caption(
        "Scans all 4 memory tickers under your unified 1x base velocity assumption (automatically scaling 2x leverage & variance drag for `RAM` and `MUU`) to identify the single highest-ROI LEAPS Spread or Naked Call across the entire memory complex."
    )

    if st.button("🚀 Run Live Cross-Ticker Memory Scan (MU, DRAM, RAM, MUU)", type="primary"):
        summary_rows = []
        combined_top_trades = []

        prog = st.progress(0.0, text="Scanning memory complex option chains...")
        tickers_list = list(MEMORY_TICKERS.keys())

        for idx, sym in enumerate(tickers_list):
            prog.progress((idx + 1) / len(tickers_list), text=f"Analyzing {sym} ({MEMORY_TICKERS[sym]['name']})...")
            try:
                s_snap = fetch_underlying_snapshot(sym)
                s_calls, s_src = get_enriched_calls_chain(
                    symbol=sym,
                    spot=s_snap["spot"],
                    hv_30d=s_snap["hv_30d"],
                    risk_free_rate=risk_free_rate,
                    data_provider=data_provider,
                    alpaca_key=alpaca_key,
                    alpaca_secret=alpaca_secret,
                    alpaca_feed=alpaca_feed,
                )
                if s_calls.empty:
                    continue

                lev = float(MEMORY_TICKERS[sym]["leverage"])
                t_price_h = adjust_target_for_leverage_decay(
                    spot=s_snap["spot"],
                    base_return_pct=base_horizon_move_pct * lev,
                    leverage=lev,
                    annual_vol=s_snap["hv_30d"],
                    horizon_days=horizon_days,
                    apply_decay=apply_leverage_decay,
                )

                s_cands, s_fb = build_leaps_and_spreads_candidates(
                    df_calls=s_calls,
                    symbol=sym,
                    spot=s_snap["spot"],
                    leverage=lev,
                    target_price_horizon=t_price_h,
                    target_price_exp_rate_monthly=monthly_vel_pct * lev,
                    horizon_days=horizon_days,
                    dte_min=dte_range[0],
                    dte_max=dte_range[1],
                    min_roi_target=min_roi_target,
                    max_contract_cost=max_contract_cost,
                    risk_free_rate=risk_free_rate,
                    apply_leverage_decay=apply_leverage_decay,
                    use_natural_fills=use_natural_fills,
                )
                if s_cands.empty:
                    continue

                top_sp = s_cands[s_cands["Strategy"] == "Bull Call Spread"]
                top_nk = s_cands[s_cands["Strategy"] == "Naked Call"]
                best_row = s_cands.iloc[0]

                sp_roi = float(top_sp.iloc[0]["Horizon ROI (%)"]) if not top_sp.empty else np.nan
                nk_roi = float(top_nk.iloc[0]["Horizon ROI (%)"]) if not top_nk.empty else np.nan

                summary_rows.append(
                    {
                        "Ticker": sym,
                        "Type": MEMORY_TICKERS[sym]["type"],
                        "Spot ($)": round(s_snap["spot"], 2),
                        f"{horizon_days}d Target ($)": round(t_price_h, 2),
                        "Winning Strategy": best_row["Strategy"],
                        "Best Contract / Spread": best_row["Structure"],
                        "DTE": int(best_row["DTE"]),
                        "Entry Cost ($)": best_row["Contract Cost ($)"],
                        "Best Spread ROI (%)": round(sp_roi, 1) if not np.isnan(sp_roi) else None,
                        "Best Naked Call ROI (%)": round(nk_roi, 1) if not np.isnan(nk_roi) else None,
                        "Winner Horizon ROI (%)": best_row["Horizon ROI (%)"],
                        "ROI Velocity (%/Day)": best_row["ROI Velocity (%/Day)"],
                        "Velocity Score": best_row["Velocity Score"],
                    }
                )
                combined_top_trades.append(s_cands.head(8))
            except Exception as e:
                st.warning(f"Could not complete scan for {sym}: {e}")

        prog.empty()

        if summary_rows:
            df_sum = pd.DataFrame(summary_rows).sort_values("Velocity Score", ascending=False)
            st.markdown("##### 🏆 Best Vehicle & Strategy Winner Per Ticker")
            st.dataframe(df_sum, use_container_width=True)

            df_all_top = pd.concat(combined_top_trades, ignore_index=True).sort_values(
                "Velocity Score", ascending=False
            )
            st.markdown("##### 🔥 Combined Top Setups Across `MU`, `DRAM`, `RAM`, `MUU`")
            st.dataframe(
                df_all_top[["Ticker"] + display_cols],
                use_container_width=True,
                height=420,
            )

# ------------------------------------------------------------------------------
# TAB 4: CUSTOM LEAPS SPREAD VS. CALL LAB
# ------------------------------------------------------------------------------
with tab4:
    st.markdown(f"#### 🧮 Custom `{active_symbol}` LEAPS Spread vs. Naked Call Structuring Lab")
    c_exp_list = sorted(df_calls["expiration"].unique().tolist(), reverse=True)
    sel_exp = st.selectbox("Select Expiration Date", options=c_exp_list, index=0)

    exp_chain = df_calls[df_calls["expiration"] == sel_exp].sort_values("strike").reset_index(drop=True)
    strikes_avail = exp_chain["strike"].tolist()
    dte_sel = int(exp_chain["dte"].iloc[0])

    # Default long strike near ATM, short strike ~20% OTM
    atm_idx = int(np.argmin(np.abs(np.array(strikes_avail) - spot_price)))
    otm_idx = min(len(strikes_avail) - 1, atm_idx + max(1, len(strikes_avail) // 5))

    l_col1, l_col2 = st.columns(2)
    with l_col1:
        custom_k_long = st.selectbox("Long Call Strike ($)", options=strikes_avail, index=atm_idx)
    with l_col2:
        short_choices = [s for s in strikes_avail if s > custom_k_long]
        if short_choices:
            custom_k_short = st.selectbox("Short Call Strike ($) [For Spread]", options=short_choices, index=min(len(short_choices)-1, 3))
        else:
            custom_k_short = None

    row_l = exp_chain[exp_chain["strike"] == custom_k_long].iloc[0]
    c_debit_naked = float(row_l["buy_fill"] if use_natural_fills else row_l["mid"])
    T_rem_custom = max(0.0, (dte_sel - horizon_days) / 365.0)
    val_nk_h, _, _, _, _ = bs_call_price_and_greeks(
        target_price_horizon, custom_k_long, T_rem_custom, risk_free_rate, float(row_l["iv_clean"])
    )
    roi_nk_h = ((val_nk_h - c_debit_naked) / c_debit_naked) * 100.0 if c_debit_naked > 0 else 0.0

    if custom_k_short is not None:
        row_s = exp_chain[exp_chain["strike"] == custom_k_short].iloc[0]
        c_credit_short = float(row_s["sell_fill"] if use_natural_fills else row_s["mid"])
        c_debit_spread = max(0.05, c_debit_naked - c_credit_short)
        c_width = custom_k_short - custom_k_long

        val_s_h, _, _, _, _ = bs_call_price_and_greeks(
            target_price_horizon, custom_k_short, T_rem_custom, risk_free_rate, float(row_s["iv_clean"])
        )
        val_sp_h = float(np.clip(val_nk_h - val_s_h, 0.0, c_width))
        roi_sp_h = ((val_sp_h - c_debit_spread) / c_debit_spread) * 100.0

        lab_df = pd.DataFrame(
            [
                {
                    "Metric": "Structure",
                    "Bull Call Debit Spread": f"{sel_exp} ${custom_k_long:g}/${custom_k_short:g}C ({dte_sel} DTE)",
                    "Naked Long Call": f"{sel_exp} ${custom_k_long:g}C ({dte_sel} DTE)",
                },
                {
                    "Metric": "Entry Cost Per Contract",
                    "Bull Call Debit Spread": f"${c_debit_spread * 100:,.0f} (${c_debit_spread:.2f}/sh)",
                    "Naked Long Call": f"${c_debit_naked * 100:,.0f} (${c_debit_naked:.2f}/sh)",
                },
                {
                    "Metric": "Expiration Breakeven",
                    "Bull Call Debit Spread": f"${custom_k_long + c_debit_spread:,.2f} ({((custom_k_long + c_debit_spread - spot_price)/spot_price)*100:+.1f}%)",
                    "Naked Long Call": f"${custom_k_long + c_debit_naked:,.2f} ({((custom_k_long + c_debit_naked - spot_price)/spot_price)*100:+.1f}%)",
                },
                {
                    "Metric": f"Projected Value in {horizon_days}d @ ${target_price_horizon:.2f}",
                    "Bull Call Debit Spread": f"${val_sp_h * 100:,.0f} (${val_sp_h:.2f}/sh)",
                    "Naked Long Call": f"${val_nk_h * 100:,.0f} (${val_nk_h:.2f}/sh)",
                },
                {
                    "Metric": f"Projected {horizon_days}d Horizon ROI (%)",
                    "Bull Call Debit Spread": f"{roi_sp_h:+.1f}%",
                    "Naked Long Call": f"{roi_nk_h:+.1f}%",
                },
                {
                    "Metric": "Net Delta / Daily Theta",
                    "Bull Call Debit Spread": f"{(row_l['delta'] - row_s['delta']):.2f}Δ / ${(row_l['theta'] - row_s['theta'])*100:.2f}/day",
                    "Naked Long Call": f"{row_l['delta']:.2f}Δ / ${row_l['theta']*100:.2f}/day",
                },
            ]
        )
        st.table(lab_df)
