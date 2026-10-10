import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import analysis as an

st.set_page_config(page_title="Does Diversification Fail in a Crash?", layout="wide")

st.title("Does diversification fail in a crash?")
st.caption("Correlation & volatility of Indian large-caps across market regimes")

# ---------- Sidebar ----------
st.sidebar.header("Settings")
all_tickers = list(an.SECTORS.keys())
tickers = st.sidebar.multiselect(
    "Stocks (NSE)", all_tickers, default=all_tickers,
    format_func=lambda t: f"{t.replace('.NS', '')}  ({an.SECTORS[t]})",
)
data_start = st.sidebar.date_input("Data start", pd.Timestamp("2016-01-01"))
roll_window = st.sidebar.slider("Rolling correlation window (days)", 20, 120, 60, 10)

st.sidebar.subheader("Market regimes")
periods = {}
for name, (s, e) in an.DEFAULT_PERIODS.items():
    c1, c2 = st.sidebar.columns(2)
    ps = c1.text_input(f"{name} start", s, key=f"{name}s")
    pe = c2.text_input("end", e, key=f"{name}e")
    periods[name] = (ps, pe)

if len(tickers) < 3:
    st.warning("Select at least 3 stocks.")
    st.stop()


# ---------- Data ----------
@st.cache_data(show_spinner="Downloading prices from Yahoo Finance...", ttl=24 * 3600)
def load_prices(tickers, start):
    raw = yf.download(list(tickers), start=str(start), auto_adjust=True, progress=False)
    prices = raw["Close"] if "Close" in raw else raw
    return prices.dropna(axis=1, how="all")


prices = load_prices(tuple(tickers), data_start)
missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.warning(f"No data returned for: {', '.join(missing)}. They were skipped.")
prices = prices[an.order_by_sector(prices.columns)]
returns = an.log_returns(prices)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Headline finding", "Heatmaps by regime", "Volatility", "Method & limitations"]
)

# --- Tab 1: headline
with tab1:
    st.subheader("Average correlation by regime")
    stats = {}
    for name, (s, e) in periods.items():
        sub = an.slice_period(returns, s, e)
        if len(sub) >= 10:
            stats[name] = an.avg_pairwise_corr(sub)

    if stats:
        base_name = list(stats.keys())[0]
        cols = st.columns(len(stats))
        for col, (name, val) in zip(cols, stats.items()):
            delta = None if name == base_name else f"{val - stats[base_name]:+.2f} vs {base_name}"
            col.metric(name, f"{val:.2f}", delta)

        if "COVID crash" in stats and base_name != "COVID crash":
            st.info(
                f"**Finding:** average pairwise correlation rose from "
                f"**{stats[base_name]:.2f}** ({base_name}) to **{stats['COVID crash']:.2f}** "
                f"during the COVID crash. Stocks that normally behave differently all fell together, "
                f"so diversification protected less exactly when it was needed most."
            )

    roll = an.rolling_avg_corr(returns, roll_window)
    fig = go.Figure(go.Scatter(x=roll.index, y=roll.values, mode="lines", name="Avg correlation"))
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]
    for (name, (s, e)), c in zip(periods.items(), colors):
        fig.add_vrect(x0=s, x1=e, fillcolor=c, opacity=0.15, line_width=0,
                      annotation_text=name, annotation_position="top left")
    fig.update_layout(
        title=f"Rolling {roll_window}-day average pairwise correlation",
        yaxis_title="Average correlation", height=450, margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: heatmaps
with tab2:
    st.subheader("Correlation matrices: calm vs stress")
    names = list(periods.keys())
    chosen = st.multiselect("Regimes to compare", names, default=names[:2])
    if chosen:
        cols = st.columns(len(chosen))
        for col, name in zip(cols, chosen):
            s, e = periods[name]
            sub = an.slice_period(returns, s, e)
            if len(sub) < 10:
                col.warning(f"Not enough data for {name}")
                continue
            corr = sub.corr()
            corr.index = [t.replace(".NS", "") for t in corr.index]
            corr.columns = corr.index
            f = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                          title=f"{name} (avg {an.avg_pairwise_corr(sub):.2f})", aspect="auto")
            f.update_layout(height=520, margin=dict(t=50))
            col.plotly_chart(f, use_container_width=True)
    st.caption("Stocks are ordered by sector, so same-sector blocks appear along the diagonal.")

# --- Tab 3: volatility
with tab3:
    st.subheader("Volatility")
    vol_window = st.slider("Rolling volatility window (days)", 10, 90, 30, 5)
    rv = an.rolling_vol(returns, vol_window).dropna(how="all")
    pick = st.multiselect(
        "Stocks to plot", list(rv.columns),
        default=list(rv.columns)[:4], format_func=lambda t: t.replace(".NS", ""),
    )
    if pick:
        f = px.line(rv[pick], title=f"Rolling {vol_window}-day annualised volatility")
        f.update_layout(height=420, yaxis_title="Annualised volatility", legend_title="")
        st.plotly_chart(f, use_container_width=True)

    vol = an.annualised_vol(returns).sort_values()
    vol.index = [t.replace(".NS", "") for t in vol.index]
    f2 = px.bar(vol, orientation="h", title="Full-period annualised volatility (low to high)")
    f2.update_layout(height=500, showlegend=False, xaxis_title="Annualised volatility", yaxis_title="")
    st.plotly_chart(f2, use_container_width=True)

# --- Tab 4: method
with tab4:
    st.subheader("Method")
    st.markdown("""
- **Data:** daily adjusted close prices from Yahoo Finance (`yfinance`), NSE tickers.
- **Returns:** daily log returns.
- **Average pairwise correlation:** the mean of all off-diagonal entries of the correlation matrix (each stock pair counted once).
- **Volatility:** standard deviation of daily returns x sqrt(252), i.e. annualised.
- **Regimes:** user-editable date windows in the sidebar.

### Limitations (read before drawing conclusions)
- The COVID window is only about 28 trading days, so correlation estimates are **noisy**.
- 19 hand-picked large caps are a small, **survivorship-biased** sample (today's winners).
- Correlation is **linear** and ignores tail dependence; it can understate co-crashes.
- Crisis windows were chosen with hindsight.
- Yahoo Finance data can contain errors; this is for learning, not investment advice.
""")