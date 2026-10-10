# Does Diversification Fail in a Crash?

An interactive study of correlation and volatility across Indian large-cap stocks in calm vs. stressed markets.

**Live app:** (link goes here after you deploy)

## Question
When markets crash, do stocks that normally move independently start moving together?

## Key result
Average pairwise correlation across 19 NSE stocks rose from **0.10** in a calm period (2017) to **0.65** during the COVID crash (Feb-Mar 2020). During the 2022 selloff it was **0.34**. In the crash, stocks from unrelated sectors all moved together, so diversification offered much less protection.


## Approach
- Daily adjusted prices for 19 NSE stocks across 7 sectors (Yahoo Finance via yfinance)
- Daily log returns
- Average pairwise correlation for each market regime
- Rolling 60-day average correlation with regimes highlighted
- Sector-ordered correlation heatmaps and rolling annualised volatility

## Run locally
pip install -r requirements.txt
python -m streamlit run app.py

## Limitations
- The COVID window is about 28 trading days, so the estimate is noisy
- 19 hand-picked large caps: small sample with survivorship bias
- Correlation is linear and misses tail dependence
- Crisis windows were chosen with hindsight
- Educational project, not investment advice