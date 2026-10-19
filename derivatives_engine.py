import numpy as np
import scipy.stats as si
from scipy.optimize import brentq
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

class OptionContract:
    """Dataclass representing a standard option contract."""
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.is_call = is_call
        self.phi = 1 if is_call else -1

class PricingEngine(ABC):
    """Abstract Base Class for all pricing engines to enforce consistent API."""
    def __init__(self, contract: OptionContract):
        self.contract = contract

    @abstractmethod
    def price(self) -> float:
        pass

class BlackScholesEngine(PricingEngine):
    def _d1_d2(self) -> Tuple[float, float]:
        c = self.contract
        d1 = (np.log(c.S / c.K) + (c.r + 0.5 * c.sigma ** 2) * c.T) / (c.sigma * np.sqrt(c.T))
        d2 = d1 - c.sigma * np.sqrt(c.T)
        return d1, d2

    def price(self) -> float:
        c = self.contract
        d1, d2 = self._d1_d2()
        return c.phi * (c.S * si.norm.cdf(c.phi * d1) - c.K * np.exp(-c.r * c.T) * si.norm.cdf(c.phi * d2))

    def analytical_greeks(self) -> Dict[str, float]:
        """Calculates exact closed-form Greeks."""
        c = self.contract
        d1, d2 = self._d1_d2()
        
        delta = si.norm.cdf(d1) if c.is_call else si.norm.cdf(d1) - 1
        gamma = si.norm.pdf(d1) / (c.S * c.sigma * np.sqrt(c.T))
        vega = c.S * si.norm.pdf(d1) * np.sqrt(c.T)
        
        theta_part1 = -(c.S * si.norm.pdf(d1) * c.sigma) / (2 * np.sqrt(c.T))
        if c.is_call:
            theta = theta_part1 - c.r * c.K * np.exp(-c.r * c.T) * si.norm.cdf(d2)
            rho = c.K * c.T * np.exp(-c.r * c.T) * si.norm.cdf(d2)
        else:
            theta = theta_part1 + c.r * c.K * np.exp(-c.r * c.T) * si.norm.cdf(-d2)
            rho = -c.K * c.T * np.exp(-c.r * c.T) * si.norm.cdf(-d2)

        return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

class BinomialLatticeEngine(PricingEngine):
    def price(self, steps: int = 1000, is_american: bool = True) -> float:
        """Highly optimized vectorized Cox-Ross-Rubinstein tree."""
        c = self.contract
        dt = c.T / steps
        u = np.exp(c.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(c.r * dt) - d) / (u - d)
        discount = np.exp(-c.r * dt)

        # Vectorized terminal nodes
        asset_prices = c.S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1, 1))
        values = np.maximum(0, c.phi * (asset_prices - c.K))

        # Backward induction using NumPy slicing (avoiding slow Python loops)
        for i in range(steps - 1, -1, -1):
            asset_prices = asset_prices[:-1] / u
            values = discount * (p * values[:-1] + (1 - p) * values[1:])
            if is_american:
                values = np.maximum(values, c.phi * (asset_prices - c.K))
                
        return values[0]

class ImpliedVolatilitySolver:
    @staticmethod
    def extract_iv(market_price: float, contract: OptionContract, tol: float = 1e-8) -> Optional[float]:
        """Robust Brent's Method for IV extraction, immune to near-zero Vega divergence."""
        def obj_func(sigma: float) -> float:
            contract.sigma = sigma
            pricer = BlackScholesEngine(contract)
            return pricer.price() - market_price

        try:
            # Bounds: 0.1% to 500% volatility
            return brentq(obj_func, 1e-4, 5.0, xtol=tol)
        except ValueError:
            return None  # Arbitrage violation in market price

# --- Example Usage ---
if __name__ == "__main__":
    contract = OptionContract(S=100, K=100, T=1.0, r=0.05, sigma=0.2, is_call=False)
    
    bs_engine = BlackScholesEngine(contract)
    tree_engine = BinomialLatticeEngine(contract)
    
    print(f"Analytical European Put: {bs_engine.price():.4f}")
    print(f"Lattice American Put:    {tree_engine.price(steps=2000, is_american=True):.4f}")
    print(f"Exact Greeks:            {bs_engine.analytical_greeks()}")