import numpy as np
import pandas as pd
import yfinance as yf
import scipy.stats as si
from arch import arch_model
from typing import Tuple

class InstitutionalRiskEngine:
    def __init__(self, ticker: str, start: str, end: str):
        self.ticker = ticker
        # Use yf.Ticker().history() instead of yf.download() to avoid MultiIndex and Adj Close issues
        raw_data = yf.Ticker(ticker).history(start=start, end=end)
        
        # 'Close' is already adjusted in the new yfinance update
        self.returns = np.log(raw_data['Close'] / raw_data['Close'].shift(1)).dropna()

    def cornish_fisher_var(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Calculates VaR and CVaR using the Cornish-Fisher Expansion.
        Adjusts standard normal quantiles for non-normal skewness and leptokurtic tails.
        """
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        skew = si.skew(self.returns)
        kurtosis = si.kurtosis(self.returns) # Excess kurtosis
        
        Z_alpha = si.norm.ppf(alpha)
        
        # Cornish-Fisher adjustment factor
        Z_cf = Z_alpha + (Z_alpha**2 - 1) * (skew / 6) + \
               (Z_alpha**3 - 3*Z_alpha) * (kurtosis / 24) - \
               (2*Z_alpha**3 - 5*Z_alpha) * (skew**2 / 36)
               
        var_cf = mu + sigma * Z_cf
        
        # Expected Shortfall (CVaR) approximation for fat tails
        tail_returns = self.returns[self.returns <= var_cf]
        cvar_cf = tail_returns.mean() if len(tail_returns) > 0 else var_cf

        return var_cf, cvar_cf

    def asymmetric_garch_forecast(self) -> float:
        """
        Fits an EGARCH (Exponential GARCH) model which captures the 'leverage effect' 
        (negative shocks increase volatility more than positive shocks).
        """
        # Rescale for optimizer stability
        scaled_returns = self.returns * 100 
        
        # EGARCH(1,1) with Student-T distribution for heavy tails
        model = arch_model(scaled_returns, vol='EGARCH', p=1, o=1, q=1, dist='studentst')
        res = model.fit(disp='off')
        
        forecast = res.forecast(horizon=1)
        next_day_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        return next_day_vol

    def backtest_var_kupiec(self, var_estimates: pd.Series, alpha: float = 0.05) -> dict:
        """
        Kupiec's Proportion of Failures (POF) test.
        Statistically verifies if the risk model is actually accurate.
        """
        exceptions = (self.returns < var_estimates).sum()
        N = len(self.returns)
        failure_rate = exceptions / N
        
        # Likelihood Ratio test statistic
        p = alpha
        lr_pof = -2 * np.log(((1 - p)**(N - exceptions)) * (p**exceptions)) + \
                  2 * np.log(((1 - failure_rate)**(N - exceptions)) * (failure_rate**exceptions))
                  
        p_value = 1 - si.chi2.cdf(lr_pof, df=1)
        is_model_valid = p_value > 0.05 # Null hypothesis: model is correct
        
        return {
            "Expected Violations": N * alpha,
            "Actual Violations": exceptions,
            "POF p-value": p_value,
            "Model Valid (95%)": is_model_valid
        }

# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing Institutional Risk Engine...")
    engine = InstitutionalRiskEngine("^GSPC", "2018-01-01", "2024-01-01")
    
    # 1. Cornish-Fisher Non-Normal VaR
    var_99, cvar_99 = engine.cornish_fisher_var(alpha=0.01)
    print(f"\nCornish-Fisher VaR (99%):  {var_99:.4%}")
    print(f"Expected Shortfall (CVaR): {cvar_99:.4%}")
    
    # 2. Leverage-Effect Volatility Forecast
    egarch_vol = engine.asymmetric_garch_forecast()
    print(f"\nEGARCH Forecasted Next-Day Vol: {egarch_vol:.4%}")
    
    # 3. Model Backtesting (Simulating a constant Historical VaR for demonstration)
    hist_var_rolling = engine.returns.rolling(252).quantile(0.01).dropna()
    engine.returns = engine.returns.loc[hist_var_rolling.index] # Align indices
    kupiec_results = engine.backtest_var_kupiec(hist_var_rolling, alpha=0.01)
    
    print("\n--- Kupiec POF Backtest Results (99% Historical VaR) ---")
    for k, v in kupiec_results.items():
        print(f"{k}: {v}")