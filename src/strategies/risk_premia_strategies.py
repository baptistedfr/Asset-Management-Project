from dataclasses import dataclass
import numpy as np
from typing import Optional
from ..tools import FrequencyType
from scipy.stats import gmean
from .abstract_strategy import AbstractStrategy, AbstractLongShortStrategy
import pandas as pd


class FractileMomentumStrategy(AbstractStrategy):
    
    def __init__(self, rebalance_frequency : FrequencyType = FrequencyType.MONTHLY, lookback_period : float = 252, 
                 nb_fractile = 4,  n_ante = 21, mean_reverting : bool = False):
        
        super().__init__(rebalance_frequency, lookback_period)
        self.nb_fractile = nb_fractile
        self.n_ante = n_ante  # Nombre de jours exclus avant la date actuelle
        self.mean_reverting = mean_reverting

    def get_position(self, historical_data: np.ndarray, current_position: np.ndarray) -> np.ndarray:
        """
        Calcule les nouvelles positions basées sur les données historiques.

        Args:
            historical_data (np.ndarray): Données historiques des prix ou rendements (shape: [time, assets]).
            current_position (np.ndarray): Positions actuelles du portefeuille.

        Returns:
            np.ndarray: Nouvelles pondérations pour chaque actif.
        """
        # Extrait les données pertinentes pour le lookback period
        data = historical_data[-int(self.lookback_period) - 1 :-self.n_ante - 1]
        # Calcule les nouvelles pondérations en utilisant la méthode fit
        new_weights = self.fit(data)
        return new_weights

    def fit(self, data: np.ndarray) -> np.ndarray:
        """
        Calcule les pondérations basées sur le momentum fractile.

        Args:
            data (np.ndarray): Données historiques des prix ou rendements (shape: [time, assets]).

        Returns:
            np.ndarray: Pondérations calculées pour chaque actif.
        """
        # Calcul du rendement sur 12 mois - 1 mois
        momentum = data[-1]/data[0]  # Rendement sur 12 mois

        # Calcul des z-scores
        zscores = (momentum - np.nanmean(momentum)) / np.nanstd(momentum)
        zscores = np.nan_to_num(zscores, nan=0.0)
        # Utilisation de pd.qcut pour diviser les actifs en fractiles
        fractiles = pd.qcut(zscores, q=self.nb_fractile, labels=False, duplicates='drop')

        # Initialisation des pondérations
        weights = np.zeros_like(zscores)

        
        if self.mean_reverting :
            # Long sur les actifs qui sous performent
            long_fractile = np.where(fractiles == 0)[0]
        else:
            # Long sur les actifs qui surperforment
            long_fractile = np.where(fractiles == np.max(fractiles))[0]

        weights[long_fractile] = 1 / len(long_fractile)

        # # Short sur le dernier fractile (z-scores les plus élevés)
        # short_fractile = np.where(fractiles == 0)[0]
        # weights[short_fractile] = -1 / len(short_fractile)
        return weights
        


