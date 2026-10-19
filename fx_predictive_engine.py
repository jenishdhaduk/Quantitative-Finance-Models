import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings
import matplotlib.pyplot as plt

# Suppress HMMlearn deprecation warnings for cleaner output
warnings.filterwarnings("ignore")

class FXPredictiveEngine:
    """
    Advanced FX Trading Engine utilizing Hidden Markov Models for Regime Detection 
    and Random Forest Meta-Labeling for trade signal filtering.
    """
    def __init__(self, pair: str, start_date: str, end_date: str):
        self.pair = pair
        self.start_date = start_date
        self.end_date = end_date
        self.data = pd.DataFrame()

    def fetch_and_engineer_features(self):
        """Fetches FX data and engineers stationary features."""
        print(f"Fetching data for {self.pair}...")
        df = yf.download(self.pair, start=self.start_date, end=self.end_date, progress=False)
        
        # We use 'Close' for FX rates
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_5d'] = df['Returns'].rolling(window=5).std()
        df['Vol_21d'] = df['Returns'].rolling(window=21).std()
        
        # Momentum features (Moving Average Convergence Divergence approximation)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Momentum'] = df['SMA_10'] / df['SMA_50'] - 1
        
        self.data = df.dropna()
        print(f"Data engineered. Shape: {self.data.shape}")

    def detect_market_regimes(self):
        """Uses Gaussian HMM to detect latent market regimes (Trending vs. Ranging)."""
        print("Calibrating Hidden Markov Model for Regime Detection...")
        
        # Features for HMM: Returns and 21-day Volatility
        hmm_features = self.data[['Returns', 'Vol_21d']].values
        
        # Train a 2-state HMM (e.g., State 0: Low Vol/Ranging, State 1: High Vol/Trending)
        hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        hmm_model.fit(hmm_features)
        
        self.data['Regime'] = hmm_model.predict(hmm_features)
        print("Market Regimes successfully classified.")

    def primary_trading_model(self):
        """
        Base model generating initial trading signals.
        Strategy: Trend following based on SMA crossovers.
        """
        # Signal: 1 for Long, -1 for Short
        self.data['Primary_Signal'] = np.where(self.data['SMA_10'] > self.data['SMA_50'], 1, -1)
        
        # Forward return (what we actually earn if we take the trade)
        self.data['Forward_Return'] = self.data['Returns'].shift(-1)
        self.data.dropna(inplace=True)

    def meta_labeling_and_prediction(self):
        """
        Secondary ML Model (Meta-Labeling). 
        Predicts if the Primary Signal will be profitable (1) or a false positive (0).
        """
        print("Training Meta-Labeling Random Forest Classifier...")
        
        # Meta-Label: 1 if the primary signal made money, 0 otherwise
        self.data['Meta_Target'] = np.where(
            np.sign(self.data['Primary_Signal']) == np.sign(self.data['Forward_Return']), 1, 0
        )
        
        # Features for the Random Forest
        features = ['Vol_5d', 'Vol_21d', 'Momentum', 'Regime']
        X = self.data[features]
        y = self.data['Meta_Target']
        
        # TimeSeriesSplit prevents data leakage (look-ahead bias)
        tscv = TimeSeriesSplit(n_splits=5)
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        predictions = np.zeros(len(y))
        
        # Walk-forward cross-validation
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            
            rf_model.fit(X_train, y_train)
            predictions[test_idx] = rf_model.predict(X_test)
            
        self.data['Meta_Prediction'] = predictions
        
        # Final Position: Only trade if Meta-Model says the Primary Signal is correct
        self.data['Final_Position'] = self.data['Primary_Signal'] * self.data['Meta_Prediction']
        
        print("\n--- Meta-Model Classification Report ---")
        # We only evaluate on the out-of-sample splits (excluding the first training window)
        test_mask = self.data['Meta_Prediction'] > 0 
        if test_mask.sum() > 0:
             print(f"Model successfully filtered {(self.data['Meta_Prediction'] == 0).sum()} false signals.")
             
    def backtest_strategy(self):
        """Evaluates financial performance of the Meta-Labeled Strategy."""
        # Calculate Strategy Returns
        self.data['Strategy_Return'] = self.data['Final_Position'] * self.data['Forward_Return']
        
        # Cumulative Returns
        self.data['Cum_Market'] = np.exp(self.data['Forward_Return'].cumsum())
        self.data['Cum_Strategy'] = np.exp(self.data['Strategy_Return'].cumsum())
        
        # Metrics
        annualized_return = self.data['Strategy_Return'].mean() * 252
        annualized_vol = self.data['Strategy_Return'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Max Drawdown
        rolling_max = self.data['Cum_Strategy'].cummax()
        drawdown = self.data['Cum_Strategy'] / rolling_max - 1
        max_drawdown = drawdown.min()
        
        print("\n--- Hedge Fund Strategy Backtest Results ---")
        print(f"Total Return (Buy & Hold):   {self.data['Cum_Market'].iloc[-1] - 1:.2%}")
        print(f"Total Return (Meta-Labeled): {self.data['Cum_Strategy'].iloc[-1] - 1:.2%}")
        print(f"Annualized Volatility:       {annualized_vol:.2%}")
        print(f"Sharpe Ratio:                {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown:            {max_drawdown:.2%}")

    def visualize_results(self):
        """Generates institutional-grade visualizations of regimes and performance."""
        print("Generating visualizations...")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1.5, 1]})
        fig.suptitle(f"FX Meta-Labeling & Regime Detection ({self.pair})", fontsize=16)

        # 1. Price Chart with Regime Backgrounds
        ax1.plot(self.data.index, self.data['Close'], color='black', linewidth=1, label='Close Price')
        ax1.fill_between(self.data.index, self.data['Close'].min(), self.data['Close'].max(), 
                         where=self.data['Regime']==1, color='red', alpha=0.1, transform=ax1.get_xaxis_transform(), label='Regime 1 (e.g., High Vol)')
        ax1.fill_between(self.data.index, self.data['Close'].min(), self.data['Close'].max(), 
                         where=self.data['Regime']==0, color='blue', alpha=0.1, transform=ax1.get_xaxis_transform(), label='Regime 0 (e.g., Low Vol)')
        ax1.set_title("Price Action & HMM Market Regimes")
        ax1.set_ylabel("Exchange Rate")
        ax1.legend(loc='upper left')

        # 2. Cumulative Returns
        ax2.plot(self.data.index, self.data['Cum_Market'] - 1, label='Buy & Hold Baseline', color='gray', linestyle='--')
        ax2.plot(self.data.index, self.data['Cum_Strategy'] - 1, label='Meta-Labeled Strategy', color='green', linewidth=1.5)
        ax2.set_title("Cumulative Returns")
        ax2.set_ylabel("Return")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax2.legend(loc='upper left')

        # 3. Drawdowns
        rolling_max = self.data['Cum_Strategy'].cummax()
        drawdown = self.data['Cum_Strategy'] / rolling_max - 1
        ax3.fill_between(self.data.index, drawdown, 0, color='darkred', alpha=0.5)
        ax3.set_title("Strategy Drawdown")
        ax3.set_ylabel("Drawdown Depth")
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Test on EUR/USD FX Pair
    fx_engine = FXPredictiveEngine(pair="EURUSD=X", start_date="2015-01-01", end_date="2024-01-01")
    
    fx_engine.fetch_and_engineer_features()
    fx_engine.detect_market_regimes()
    fx_engine.primary_trading_model()
    fx_engine.meta_labeling_and_prediction()
    fx_engine.backtest_strategy()
