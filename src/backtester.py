from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import cached_property
import warnings
warnings.filterwarnings("ignore")
from .strategies import AbstractStrategy
from .results import Results
from .tools import FrequencyType

class Backtester:
    

    """
    Generic class to backtest strategies from assets prices & a strategy

    Args:
        data_input (DataInput) : data input object containing assets prices historic
    """

    def __init__(self, df_prices : pd.DataFrame, df_weights : pd.DataFrame = None):
        self.df_prices : pd.DataFrame = df_prices
        self.df_weights : pd.DataFrame = df_weights
        self.df_prices['Date'] = pd.to_datetime(self.df_prices['Date'], format = "%Y-%m-%d")
        self.df_weights['Date'] = pd.to_datetime(self.df_weights['Date'],format = "%Y-%m-%d")

    def run(self, 
            start_date_str : str,
            end_date_str : str,
            strategy : AbstractStrategy, 
            initial_amount : float = 1000.0, 
            fees : float = 0.001, 
            custom_name : str = None) -> Results :
        
        """Run the backtest over the asset period (& compare with the benchmark if selected)
        
        Args:
            strategy (AbstractStrategy) : instance of Strategy class with "compute_weights" method
            initial_amount (float) : initial value of the portfolio
            fees (float) : transaction fees for every portfolio weight rebalancing
            delayed_start (optional str) : possibility to start the backtest after the first date of the data input 
                                           (used in the strategies to have enough data at the beginning of the backtest)

        Returns:
            Results: A Results object containing statistics and comparison plot for the strategy (& the benchmark if selected)
        """

        """Verification of some basic instances"""

        self._valide_inputs(strategy, initial_amount, fees)

        """Get the start & end date in datetime format"""
        start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        """Get the list of tickers in the universe between the start & end date"""
        start_date, end_date = self.get_analysis_dates(self.df_prices, start_date_dt, end_date=end_date_dt, n_before = strategy.lookback_period)
        common_tickers = self.get_tickers_in_range(start_date_dt, end_date)
        
        # Get the prices & returns for the selected tickers
        df_prices = self.df_prices.loc[(self.df_prices['Date'] >= start_date) & (self.df_prices['Date'] <= end_date), ['Date'] + list(common_tickers)]
        df_returns = df_prices.iloc[:,1:].pct_change()
        all_dates = sorted(set(df_prices["Date"].to_numpy()))
    
        """Get the rebalancing dates"""
        rebalancing_dates = self._get_rebalancing_dates(df_prices, strategy.rebalance_frequency)

        # Initialisation
        strat_name = custom_name if custom_name else strategy.__class__.__name__
        strat_value = initial_amount
        weights_dict = {}
        stored_values = [strat_value]

        # Initialiser les poids pour les tickers initiaux
        actual_tickers = self.get_tickers_in_range(all_dates[strategy.lookback_period])
        weights = dict(zip(actual_tickers, [1/len(actual_tickers)] * len(actual_tickers)))
        weights_dict[all_dates[strategy.lookback_period]] = weights

        stored_values = [strat_value]
        # benchmark_returns_matrix = self.benchmark_returns
        # if benchmark_returns_matrix is not None :
        #     benchmark_value = initial_amount
        #     stored_benchmark = [benchmark_value]
        #     benchmark_returns_matrix = benchmark_returns_matrix.to_numpy()
        total_fees = 0
        for t in tqdm(range(strategy.lookback_period+1, len(df_prices)),desc=f"Running Backtesting {strat_name}"):
            current_date = all_dates[t]

            # Filtrer les rendements pour les tickers actifs
            active_returns = df_returns.loc[df_prices['Date'] == current_date, list(weights.keys())].fillna(0).squeeze().to_dict()
            
            # Calculer la valeur du portefeuille
            daily_returns = np.array([active_returns.get(ticker, 0) for ticker in weights.keys()])
            prev_weights = np.array(list(weights.values()))
            return_strat = np.dot(prev_weights, daily_returns)
            new_strat_value = strat_value * (1 + return_strat)
            
            if current_date in rebalancing_dates:
                """Get the new tickers in the universe"""
                actual_tickers = self.get_tickers_in_range(all_dates[t])
                """Use Strategy to compute new weights (Rebalancement)"""
                new_weights = strategy.get_position(df_prices.loc[df_prices['Date'] <= current_date, list(actual_tickers)].to_numpy(), prev_weights)
                new_weights = dict(zip(actual_tickers, new_weights / np.sum(new_weights)))
                """Compute transaction costs"""
                transaction_costs = self.calculate_transaction_costs(weights, new_weights, fees)
                total_fees += transaction_costs
                new_strat_value -= strat_value * transaction_costs
                
            else: 
                """Apply drift to weights"""
                new_weights = {ticker: weights[ticker] * (1 + active_returns[ticker]) for ticker in weights}
                
            # Normaliser les poids après drift
            total_weight = sum(new_weights.values())
            new_weights = {ticker: weight / total_weight for ticker, weight in new_weights.items()}

            # Stocker les résultats
            weights = new_weights
            strat_value = new_strat_value
            weights_dict[current_date] =weights
            stored_values.append(strat_value)
        
        return self.output(strat_name, stored_values, weights_dict, None, list(weights_dict.keys()), FrequencyType.DAILY )
            
    def output(self, strategy_name : str, 
               stored_values : list[float], 
               stored_weights : list[float], 
               stored_benchmark : list[float] = None, 
               dates : list = [],
               frequency : FrequencyType = None) -> Results :
        """Create the output for the strategy and its benchmark if selected
        
        Args:
            stored_values (list[float]): Value of the strategy over time
            stored_weights (list[float]): Weights of every asset in the strategy over time
            stored_benchmark (list[float]): Value of the benchmark portfolio over time
            strategy_name (str) : Name of the current strategy

        Returns:
            Results: A Results object containing statistics and comparison plot for the strategy (& the benchmark if selected)
        """

        ptf_weights = pd.DataFrame(stored_weights).T
        ptf_values = pd.Series(stored_values, index=dates)

        results_strat = Results(ptf_values=ptf_values, ptf_weights=ptf_weights, strategy_name=strategy_name, data_frequency=frequency)
        results_strat.get_statistics()
        results_strat.create_plots()

        if stored_benchmark is not None :

            benchmark_values = pd.Series(stored_benchmark, index=dates)
            results_bench = Results(ptf_values=benchmark_values, strategy_name="Benchmark", data_frequency=frequency)
            results_bench.get_statistics()
            results_bench.create_plots()
            results_strat = Results.compare_results([results_strat, results_bench])

        return results_strat
    
    @staticmethod
    def calculate_transaction_costs(old_weights: dict, new_weights: dict, fees: float) -> float:
        """
        Calcule les frais de transaction basés sur les changements de poids.

        Args:
            old_weights (dict): Poids des actifs avant le rebalancement (ticker -> poids).
            new_weights (dict): Poids des actifs après le rebalancement (ticker -> poids).
            fees (float): Taux des frais de transaction (par exemple, 0.0005 pour 0.05%).

        Returns:
            float: Coût total des transactions.
        """
        # Obtenir l'ensemble des tickers impliqués
        all_tickers = set(old_weights.keys()).union(set(new_weights.keys()))

        # Calculer les frais de transaction pour chaque ticker
        transaction_costs = fees * np.sum(
            np.abs(np.array([new_weights.get(t, 0) - old_weights.get(t, 0) for t in all_tickers]))
        )

        return transaction_costs
    
    @staticmethod
    def _get_rebalancing_dates(df_prices: pd.DataFrame, frequency: FrequencyType, custom_freq: str = None) -> list:
        """
        Repère les indices correspondant aux dates de rebalancement en fonction de la fréquence donnée.

        Args:
            df_prices (pd.DataFrame): DataFrame contenant les prix avec une colonne 'Date'.
            frequency (FrequencyType): Fréquence de rebalancement souhaitée.
            custom_freq (str, optional): Fréquence personnalisée (ex: '2M' pour bimensuel, 'Q' pour trimestriel).

        Returns:
            list: Liste des dates de rebalancement.
        """
        if not pd.api.types.is_datetime64_any_dtype(df_prices["Date"]):
            df_prices["Date"] = pd.to_datetime(df_prices["Date"])

        if frequency == FrequencyType.DAILY:
            # Toutes les dates sont incluses
            return df_prices["Date"].tolist()

        if frequency == FrequencyType.WEEKLY:
            # Rebalancement hebdomadaire
            return df_prices.groupby(df_prices["Date"].dt.to_period("W"))["Date"].max().tolist()

        if frequency == FrequencyType.MONTHLY:
            # Rebalancement mensuel
            return df_prices.groupby(df_prices["Date"].dt.to_period("M"))["Date"].max().tolist()

        if frequency == FrequencyType.QUARTERLY:
            # Rebalancement trimestriel
            return df_prices.groupby(df_prices["Date"].dt.to_period("Q"))["Date"].max().tolist()

        if custom_freq:
            # Rebalancement avec une fréquence personnalisée (ex: '2M' pour bimensuel)
            return df_prices.resample(custom_freq, on="Date")["Date"].max().dropna().tolist()

        raise ValueError("Fréquence de rebalancement non supportée.")
    
    def _valide_inputs(self, strategy : AbstractStrategy, initial_amount : int, fees : float, ):
        if not isinstance(initial_amount, (int, float)) or initial_amount <= 0:
            raise ValueError("Initial amount must be a positive number.")
        if not (0 <= fees <= 1):
            raise ValueError("Fees must be a proportion between 0 and 1.")

        if strategy.lookback_period > len(self.df_prices):
            raise ValueError("We don't have enought data to run our backtest !")

    def get_tickers_in_range(self, start_date_dt: datetime, end_date_dt: datetime = None) -> list:
        """
        Récupère tous les tickers présents dans l'indice entre deux dates.

        Args:
            start_date_dt (datetime): Date de début.
            end_date_dt (datetime, optional): Date de fin. Default: None.

        Returns:
            list: Liste des tickers valides avec suffixe " Equity".
        """

        # Dernière composition avant start_date
        closest_row_universe = self.df_weights[self.df_weights['Date'] <= start_date_dt].sort_values('Date').tail(1).iloc[:,1:]

        if closest_row_universe.empty:
            raise ValueError("Impossible de récupérer la composition de l'univers avant la start_date.")

        # Récupérer les tickers avec des poids > 0
        tickers_universe = set(
            ticker for ticker in closest_row_universe.loc[:, closest_row_universe.gt(0).iloc[0]].columns.to_list()
        )

        # Si une date de fin est spécifiée, récupérer les tickers actifs dans la plage de dates
        if end_date_dt:
            filtered_universe = self.df_weights[(self.df_weights['Date'] >= start_date_dt) & 
                                                (self.df_weights['Date'] <= end_date_dt)].iloc[:,1:]
            
            if not filtered_universe.empty:
                active_tickers = set(
                    ticker for ticker in filtered_universe.loc[:, filtered_universe.gt(0).any()].columns.to_list()
                )
                tickers_universe.update(active_tickers)

        # Vérifier les tickers communs avec les données de prix
        common_tickers = self.get_common_tickers(self.df_prices, list(tickers_universe))

        return common_tickers
    
    @staticmethod
    def get_common_tickers(df_prices : pd.DataFrame, tickers_universe : list) -> set:
        df_prices = df_prices.dropna(axis=1, how = 'all') 
        common_tickers = set(tickers_universe)
        return common_tickers.intersection(set(df_prices.columns))
    
    @staticmethod
    def get_analysis_dates(price_df: pd.DataFrame, computation_date: datetime, 
                           n_before: int = 253,
                           end_date: datetime = None):
        """
        Récupère les dates de début et de fin pour l'analyse en fonction des jours de marché.

        Args:
            price_df (pd.DataFrame): DataFrame contenant les prix avec une colonne 'Date'.
            computation_date (datetime): Date de référence pour l'analyse.
            n_before (int, optional): Nombre de jours de marché avant la computation_date. Par défaut 253.
            end_date (datetime, optional): Date de fin spécifique.
        Returns:
            tuple(datetime, datetime): (start_date, end_date)
        """
        price_dates = price_df['Date'].sort_values().unique()

        # Trouver les dates avant computation_date
        past_dates = price_dates[price_dates < computation_date]
        if len(past_dates) < n_before:
            raise ValueError(f"Pas assez de données avant {computation_date}. ({len(past_dates)} jours disponibles)")

        start_date = past_dates[-n_before]  # n_before jours avant

        if end_date:
            # Vérifier que end_date est dans le range
            if end_date <= computation_date:
                raise ValueError("end_date doit être postérieure à computation_date.")
            valid_future_dates = price_dates[price_dates > computation_date]
            if len(valid_future_dates) == 0 or end_date > valid_future_dates[-1]:
                raise ValueError("end_date dépasse les données disponibles.")
            
        return start_date, end_date