# Quantitative Finance & Risk Models
This repository contains production-grade Python implementations of quantitative finance models, focusing on derivatives pricing and institutional econometric risk estimation.

### 1. Advanced Derivatives Pricing Engine (`derivatives_engine.py`)
* **Object-Oriented Architecture:** Abstract Base Classes coupling `OptionContract` data models with interchangeable pricing engines.
* **Pricing Models:** Implemented closed-form Black-Scholes-Merton and highly optimized, vectorized Cox-Ross-Rubinstein (CRR) binomial lattice trees for American and European options.
* **Greeks & Sensitivities:** Exact closed-form analytical Greeks and finite-difference stochastic sensitivities.
* **Implied Volatility Extraction:** Robust Brent's method root-finding algorithm to invert market option chains, immune to near-zero Vega divergence.

### 2. Institutional Risk Metrics Engine (`risk_metrics_engine.py`)
* **Cornish-Fisher Expansion:** Adjusts Value at Risk (VaR) and Expected Shortfall (CVaR) quantiles for non-normal skewness and leptokurtic tails (fat tails).
* **Asymmetric Volatility Forecasting:** Calibrated EGARCH (Exponential GARCH) time-series models with Student-T distributions to capture the 'leverage effect' in equity market drawdowns.
* **Model Validation & Backtesting:** Implemented Kupiec's Proportion of Failures (POF) Likelihood Ratio test to statistically validate VaR model accuracy against historical actuals.

### 3. FX Predictive Engine (Regime Detection & Meta-Labeling)
* **Hidden Markov Models (HMM):** Calibrated Gaussian HMMs to detect latent market regimes (Trending vs. Ranging) using non-stationary volatility and return distributions.
* **Meta-Labeling ML Architecture:** Engineered a Random Forest classifier to filter false-positive trading signals using Time-Series walk-forward cross-validation, strictly preventing data leakage.
* **Algorithmic Backtesting:** Evaluated signal efficacy, generating strategy Sharpe Ratios and Max Drawdown metrics against buy-and-hold FX benchmarks.
