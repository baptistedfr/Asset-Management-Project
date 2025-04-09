from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional
from typing import Optional, Callable
from functools import wraps

class AbstractStrategy(ABC):
    
   

    def __init__(self, rebalance_frequency = None, lookback_period : float = 252):
        """Abstract class to represent a strategy used in backtesting.
            Args:
                rebalance_frequency : FrequencyType : Choice of the rebalancing frequence of the strategy. Default Value is Monthly rebalancement
                lookback_period : float : The historical period (%year) considered by the strategy to calculate indicators or make decisions. Default value is one year
        """
        self.rebalance_frequency = rebalance_frequency
        self.lookback_period = lookback_period

    @abstractmethod
    def get_position(self, historical_data : np.ndarray[float], 
                     current_position: np.ndarray[float], 
                     benchmark: Optional[np.ndarray] = None,
                     sector : Optional[dict] = None) -> np.ndarray[float]:
        """
        Mandatory method to be implemented by all strategies.
        Calculates the new position based on historical data and the current position.

        Args:
            historical_data : Historical data required for decision-making (e.g., prices, returns, etc.).
            current_position : The current position of the strategy (e.g., current asset weights).
        
        Returns:
            The new positions in a numpy array
        """
        pass

    def fit(self, data):
        """
        Optional method.
        Can be used to train or calibrate the strategy (does nothing by default).
        
        Args:
            data : Data required to train the strategy.
        """
        pass

    def compute_na(self, weights, returns):
        """
        Method to handle NaN values by setting weights to 0 where returns are NaN and adjusts the weights correspondously
        """
        weights[np.isnan(returns)] = 0
        return weights

    def valid_assets_data(self, data):
        """
        Method to filter the data for valid assets (at least one non-NaN value)
        """
        valid_assets = ~np.any(np.isnan(data), axis=0)
        return data[:, valid_assets], valid_assets


@dataclass 
class AbstractLongShortStrategy(AbstractStrategy, ABC):
    """
    Abstract class for Long/Short strategies.
    """
    is_LS_strategy: Optional[bool] = False


# Decorator definition
def build_strategy(func: Callable) -> AbstractStrategy:
    """
    Decorator to transform a function into a strategy compatible with AbstractStrategy.

    Args:
        func: A function implementing the logic for `get_position`.

    Returns:
        A callable AbstractStrategy instance with the provided `get_position` logic.
    """
    class DecoratedStrategy(AbstractStrategy):
        def get_position(self, historical_data: np.ndarray, current_position: np.ndarray) -> np.ndarray:
            return func(historical_data, current_position)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return DecoratedStrategy()  # Return an instance of the decorated strategy
    
    return wrapper()