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

    def get_position(self, historical_data: np.ndarray, current_position: np.ndarray, benchmark: Optional[np.ndarray] = None) -> np.ndarray:
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
        


class IdiosyncraticMomentumStrategy(AbstractStrategy):
    def __init__(self, rebalance_frequency: FrequencyType = FrequencyType.MONTHLY, lookback_period: float = 252, 
                 nb_fractile=4, n_ante=21, mean_reverting: bool = False):
        """
        Initialise la stratégie de momentum idiosyncratique.

        Args:
            rebalance_frequency (FrequencyType): Fréquence de rééquilibrage.
            lookback_period (float): Période de lookback pour le calcul du momentum.
            nb_fractile (int): Nombre de fractiles pour classer les actifs.
            n_ante (int): Nombre de jours exclus avant la date actuelle.
            mean_reverting (bool): Active le momentum mean-reverting.
        """
        super().__init__(rebalance_frequency, lookback_period)
        self.nb_fractile = nb_fractile
        self.n_ante = n_ante
        self.mean_reverting = mean_reverting

    def get_position(self, historical_data: np.ndarray, current_position: np.ndarray,  benchmark: np.ndarray, ) -> np.ndarray:
        """
        Calcule les nouvelles positions basées sur les données historiques et le benchmark.

        Args:
            historical_data (np.ndarray): Données historiques des prix ou rendements (shape: [time, assets]).
            benchmark (np.ndarray): Données historiques des rendements du benchmark (shape: [time]).
            current_position (np.ndarray): Positions actuelles du portefeuille.

        Returns:
            np.ndarray: Nouvelles pondérations pour chaque actif.
        """
        # Extrait les données pertinentes pour le lookback period
        data = historical_data[-int(self.lookback_period) - 1 :-self.n_ante - 1]
        benchmark_data = benchmark[-int(self.lookback_period) - 1 :-self.n_ante - 1]

        # Calcule les nouvelles pondérations en utilisant la méthode fit
        new_weights = self.fit(data, benchmark_data)
        return new_weights

    def fit(self, data: np.ndarray, benchmark: np.ndarray) -> np.ndarray:
        """
        Calcule les pondérations basées sur le momentum idiosyncratique.

        Args:
            data (np.ndarray): Données historiques des prix (shape: [time, assets]).
            benchmark (np.ndarray): Données historiques du benchmark (shape: [time]).

        Returns:
            np.ndarray: Pondérations calculées pour chaque actif.
        """
        # Vérification des dimensions
        if data.shape[0] != benchmark.shape[0]:
            raise ValueError("Les dimensions de 'data' et 'benchmark' doivent correspondre.")

        # Calcul des rendements journaliers
        asset_returns = data[1:] / data[:-1] - 1
        benchmark_returns = benchmark[1:] / benchmark[:-1] - 1
        benchmark_returns = benchmark_returns.ravel()

        num_assets = asset_returns.shape[1]
        idiosyncratic_returns = np.full_like(asset_returns, np.nan)

        # Ajout d'une constante pour estimer alpha et beta
        X = np.column_stack((np.ones_like(benchmark_returns), benchmark_returns))

        for i in range(num_assets):
            y = asset_returns[:, i]

            # Vérifier si les données sont valides
            if np.isnan(y).any() or np.isnan(benchmark_returns).any():
                continue

            # Estimation des coefficients (alpha et beta)
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = coef

            # Calcul des rendements idiosyncratiques
            residuals = y - (alpha + beta * benchmark_returns)
            idiosyncratic_returns[:, i] = residuals

        # Calcul du momentum idiosyncratique
        momentum = idiosyncratic_returns[-1]/idiosyncratic_returns[0]
        # Gestion des cas où tous les rendements sont NaN
        if np.all(np.isnan(momentum)):
            return np.zeros(data.shape[1])

        # Calcul des z-scores
        zscores = (momentum - np.nanmean(momentum)) / np.nanstd(momentum)
        zscores = np.nan_to_num(zscores, nan=0.0)

        # Découpage en fractiles
        fractiles = pd.qcut(zscores, q=self.nb_fractile, labels=False, duplicates='drop')
        weights = np.zeros_like(zscores)

        if self.mean_reverting:
            long_fractile = np.where(fractiles == 0)[0]
        else:
            long_fractile = np.where(fractiles == np.max(fractiles))[0]

        if len(long_fractile) > 0:
            weights[long_fractile] = 1 / len(long_fractile)

        return weights