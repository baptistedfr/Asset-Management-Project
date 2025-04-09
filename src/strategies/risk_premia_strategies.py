from dataclasses import dataclass
import numpy as np
from typing import Optional
from ..tools import FrequencyType
from scipy.stats import gmean
from .abstract_strategy import AbstractStrategy, AbstractLongShortStrategy
import pandas as pd


class FractileMomentumStrategy(AbstractStrategy):
    
    def __init__(self, rebalance_frequency : FrequencyType = FrequencyType.MONTHLY, lookback_period : float = 252, 
                 nb_fractile = 4,  n_ante = 21, mean_reverting : bool = False, is_segmentation_sectorial : bool = False, df_sector = pd.DataFrame()):
        
        super().__init__(rebalance_frequency, lookback_period)
        self.nb_fractile = nb_fractile
        self.n_ante = n_ante  # Nombre de jours exclus avant la date actuelle
        self.mean_reverting = mean_reverting
        self.is_segmentation_sectorial = is_segmentation_sectorial
        self.df_sector = df_sector  # DataFrame contenant les secteurs des actifs

    def get_position(self, historical_data: np.ndarray, 
                     current_position: np.ndarray, 
                     benchmark: Optional[np.ndarray] = None,
                     sector_repartition : dict = None,
                     tickers : list = None) -> np.ndarray:
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
        new_weights = self.fit(data, tickers, sector_repartition)
        return new_weights

    def fit(self, data: np.ndarray, tickers: list = None, sector_repartition : dict = None) -> np.ndarray:
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
        weights = np.zeros_like(zscores)

        if not self.is_segmentation_sectorial or self.df_sector.empty or tickers is None:
            # Comportement normal sans segmentation
            fractiles = pd.qcut(zscores, q=self.nb_fractile, labels=False, duplicates='drop')
            if self.mean_reverting:
                selected = np.where(fractiles == 0)[0]
            else:
                selected = np.where(fractiles == np.max(fractiles))[0]
            weights[selected] = 1 / len(selected) if len(selected) > 0 else 0
            return weights
    
        # === Segmentation sectorielle ===
        df_sector_map = self.df_sector.set_index('Ticker')['Secteur'].to_dict()
        ticker_sector = {ticker: df_sector_map.get(ticker, 'Unknown') for ticker in tickers}
        sector_groups = {}

        for i, ticker in enumerate(tickers):
            sector = ticker_sector[ticker]
            sector_groups.setdefault(sector, []).append(i)

        # Répartition uniforme des poids par secteur (peut être modifié)
        sector_weights = {s: sector_repartition.get(s, 0) for s in sector_groups}

        for sector, indices in sector_groups.items():
            sector_z = zscores[indices]
            if len(sector_z) == 0:
                continue
            try:
                sector_fractiles = pd.qcut(sector_z, q=self.nb_fractile, labels=False, duplicates='drop')
            except ValueError:
                # Pas assez de valeurs distinctes pour qcut
                continue

            if self.mean_reverting:
                selected = [indices[i] for i, f in enumerate(sector_fractiles) if f == 0]
            else:
                selected = [indices[i] for i, f in enumerate(sector_fractiles) if f == np.max(sector_fractiles)]

            if selected:
                per_asset_weight = sector_weights[sector] / len(selected)
                for idx in selected:
                    weights[idx] = per_asset_weight

        return weights
        


class IdiosyncraticMomentumStrategy(AbstractStrategy):
    def __init__(self, rebalance_frequency: FrequencyType = FrequencyType.MONTHLY, lookback_period: float = 252,
                 nb_fractile=4, n_ante=21, mean_reverting: bool = False,
                 is_segmentation_sectorial: bool = False, df_sector: pd.DataFrame = pd.DataFrame()):
        
        super().__init__(rebalance_frequency, lookback_period)
        self.nb_fractile = nb_fractile
        self.n_ante = n_ante
        self.mean_reverting = mean_reverting
        self.is_segmentation_sectorial = is_segmentation_sectorial
        self.df_sector = df_sector

    def get_position(self, historical_data: np.ndarray, current_position: np.ndarray, benchmark: np.ndarray,
                     tickers: list = None, sector_repartition: dict = None) -> np.ndarray:
        data = historical_data[-int(self.lookback_period) - 1: -self.n_ante - 1]
        benchmark_data = benchmark[-int(self.lookback_period) - 1: -self.n_ante - 1]
        return self.fit(data, benchmark_data, tickers, sector_repartition)

    def fit(self, data: np.ndarray, benchmark: np.ndarray,
            tickers: list = None, sector_repartition: dict = None) -> np.ndarray:
        
        if data.shape[0] != benchmark.shape[0]:
            raise ValueError("Mismatch entre les dimensions des données assets et benchmark.")

        asset_returns = data[1:] / data[:-1] - 1
        benchmark_returns = benchmark[1:] / benchmark[:-1] - 1
        benchmark_returns = benchmark_returns.ravel()

        num_assets = asset_returns.shape[1]
        idiosyncratic_returns = np.full_like(asset_returns, np.nan)
        X = np.column_stack((np.ones_like(benchmark_returns), benchmark_returns))

        for i in range(num_assets):
            y = asset_returns[:, i]
            if np.isnan(y).any():
                continue
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = coef
            residuals = y - (alpha + beta * benchmark_returns)
            idiosyncratic_returns[:, i] = residuals

        cum_returns = np.nancumprod(1 + idiosyncratic_returns, axis=0) - 1
        momentum = cum_returns[-1]
        if np.all(np.isnan(momentum)):
            return np.zeros(num_assets)

        zscores = (momentum - np.nanmean(momentum)) / np.nanstd(momentum)
        zscores = np.nan_to_num(zscores, nan=0.0)
        weights = np.zeros_like(zscores)

        # === Si segmentation sectorielle ===
        if self.is_segmentation_sectorial and not self.df_sector.empty and tickers is not None:
            df_sector_map = self.df_sector.set_index('Ticker')['Secteur'].to_dict()
            ticker_sector = {ticker: df_sector_map.get(ticker, 'Unknown') for ticker in tickers}
            sector_groups = {}
            for i, ticker in enumerate(tickers):
                sector = ticker_sector[ticker]
                sector_groups.setdefault(sector, []).append(i)


            sector_weights = {s: sector_repartition.get(s, 0) for s in sector_groups}

            for sector, indices in sector_groups.items():
                sector_z = zscores[indices]
                if len(sector_z) == 0:
                    continue
                try:
                    sector_fractiles = pd.qcut(sector_z, q=self.nb_fractile, labels=False, duplicates='drop')
                except ValueError:
                    continue

                if self.mean_reverting:
                    selected = [indices[i] for i, f in enumerate(sector_fractiles) if f == 0]
                else:
                    selected = [indices[i] for i, f in enumerate(sector_fractiles) if f == np.max(sector_fractiles)]

                if selected:
                    per_asset_weight = sector_weights[sector] / len(selected)
                    for idx in selected:
                        weights[idx] = per_asset_weight
        else:
            fractiles = pd.qcut(zscores, q=self.nb_fractile, labels=False, duplicates='drop')
            if self.mean_reverting:
                long_fractile = np.where(fractiles == 0)[0]
            else:
                long_fractile = np.where(fractiles == np.max(fractiles))[0]

            if len(long_fractile) > 0:
                weights[long_fractile] = 1 / len(long_fractile)

        return weights