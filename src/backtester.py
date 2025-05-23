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

    def __init__(self, df_prices: pd.DataFrame, df_weights: pd.DataFrame, df_benchmark: pd.DataFrame = None, df_sector = None):
        """
        Initialise le Backtester avec les données nécessaires.

        Args:
            df_prices (pd.DataFrame): DataFrame contenant les prix des actifs avec une colonne 'Date'.
            df_weights (pd.DataFrame): DataFrame contenant les poids des actifs avec une colonne 'Date'.
            df_benchmark (pd.DataFrame, optional): DataFrame contenant les données du benchmark avec une colonne 'Date'. Default: None.
        """
        # Vérification des colonnes nécessaires
        required_columns = ['Date']
        for df, name in [(df_prices, "df_prices"), (df_weights, "df_weights"), (df_benchmark, "df_benchmark")]:
            if df is not None and not all(col in df.columns for col in required_columns):
                raise ValueError(f"{name} doit contenir une colonne 'Date'.")

        # Conversion des dates et définition des index
        self.df_prices = self._prepare_dataframe(df_prices, "df_prices")
        self.df_weights = self._prepare_dataframe(df_weights, "df_weights")
        self.df_benchmark = self._prepare_dataframe(df_benchmark, "df_benchmark") if df_benchmark is not None else None
        self.df_sector = df_sector if df_sector is not None else None
        
    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Prépare un DataFrame en vérifiant la colonne 'Date', en la convertissant en datetime et en la définissant comme index.

        Args:
            df (pd.DataFrame): Le DataFrame à préparer.
            name (str): Nom du DataFrame (pour les messages d'erreur).

        Returns:
            pd.DataFrame: DataFrame préparé avec 'Date' comme index.
        """
        if 'Date' not in df.columns:
            raise ValueError(f"{name} doit contenir une colonne 'Date'.")
        
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d", errors='coerce')
        if df['Date'].isna().any():
            raise ValueError(f"{name} contient des dates invalides.")
        
        df.set_index('Date', inplace=True)
        return df

    def run(self, 
            start_date_str : str,
            end_date_str : str,
            strategy : AbstractStrategy, 
            initial_amount : float = 1000.0, 
            fees : float = 0.001, 
            custom_name : str = None,
            recompute_benchmark : bool = False) -> Results :
        
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
        df_prices = self.df_prices.loc[(self.df_prices.index >= start_date) & (self.df_prices.index <= end_date), list(common_tickers)]
        df_returns = df_prices.pct_change()
        
        all_dates = sorted(set(df_prices.index.to_numpy()))
    
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

        stored_benchmark = None
        if self.df_benchmark is not None :
            # benchmark_prices = self.df_benchmark.loc[(self.df_benchmark.index >= start_date) & (self.df_benchmark.index <= end_date), :].reindex(df_prices.index).fillna(method="ffill")
            # benchmark_returns1 = benchmark_prices.pct_change()
            if recompute_benchmark:
                # Recalculer les valeurs du benchmark
                stored_benchmark, sector_weight = self._compute_benchmark_values(
                    df_returns=df_returns,
                    start_date=start_date,
                    end_date=end_date,
                    all_dates=all_dates,
                    initial_amount=initial_amount
                )
            else:
                stored_benchmark = pd.read_excel("data/benchmark_recompute.xlsx", index_col=0, parse_dates=True).loc[all_dates[0]:]
                sector_weight = pd.read_excel("data/sector_weights.xlsx", index_col=0, parse_dates=True).loc[all_dates[0]:]
        
        total_fees = 0
        for t in tqdm(range(strategy.lookback_period+1, len(df_prices)),desc=f"Running Backtesting {strat_name}"):
            current_date = all_dates[t]

            # Filtrer les rendements pour les tickers actifs
            active_returns = df_returns.loc[df_returns.index == current_date, list(weights.keys())].fillna(0).squeeze().to_dict()
            
            # Calculer la valeur du portefeuille
            daily_returns = np.array([active_returns.get(ticker, 0) for ticker in weights.keys()])
            prev_weights = np.array(list(weights.values()))
            return_strat = np.dot(prev_weights, daily_returns)
            new_strat_value = strat_value * (1 + return_strat)

            if current_date in rebalancing_dates:
                """Get the new tickers in the universe"""
                actual_tickers = self.get_tickers_in_range(all_dates[t])
                """Use Strategy to compute new weights (Rebalancement)"""
                new_weights = strategy.get_position(df_prices.loc[df_prices.index <= current_date, list(actual_tickers)].fillna(method="ffill").to_numpy(), 
                                                    prev_weights, 
                                                    stored_benchmark.loc[stored_benchmark.index <= current_date,:].to_numpy(),
                                                    sector_repartition = sector_weight.loc[sector_weight.index == current_date].iloc[0].to_dict(),
                                                    tickers = actual_tickers)
                
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
        
        return self.output(strategy_name = strat_name, 
                           stored_values = stored_values, 
                           stored_weights = weights_dict, 
                           stored_benchmark = stored_benchmark[stored_benchmark.index>= list(weights_dict.keys())[0]] if stored_benchmark is not None else None, 
                           dates = list(weights_dict.keys()), 
                           frequency = FrequencyType.DAILY )
            
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
        
        benchmark_values = None
        if stored_benchmark is not None :
            benchmark_values = stored_benchmark["Benchmark"]/stored_benchmark["Benchmark"].iloc[0] * ptf_values[0]

        results_strat = Results(ptf_values=ptf_values, ptf_weights=ptf_weights, 
                                strategy_name=strategy_name, data_frequency=frequency, 
                                benchmark_values= benchmark_values)
        results_strat.get_statistics()
        results_strat.create_plots()

        if stored_benchmark is not None :
            results_bench = Results(ptf_values=benchmark_values, strategy_name="Benchmark", data_frequency=frequency)
            results_bench.get_statistics()
            results_bench.create_plots()
            results_strat = Results.compare_results([results_strat, results_bench])

        return results_strat
    
    def _compute_benchmark_values(self, df_returns: pd.DataFrame, start_date, end_date, all_dates, initial_amount) -> list:
        """
        Calcule la valeur du benchmark sur la période.

        Args:
            df_returns (pd.DataFrame): Rendements des prix des actifs (index = dates, colonnes = tickers)
            start_date (datetime): Date de début de la période d'analyse
            end_date (datetime): Date de fin de la période d'analyse
            all_dates (list[datetime]): Liste de toutes les dates de la période
            initial_amount (float): Valeur initiale du portefeuille benchmark

        Returns:
            list[float]: Valeurs du benchmark au fil du temps
        """
        benchmark_value = initial_amount
        stored_benchmark = [benchmark_value]

        sector_map = dict(zip(self.df_sector['Ticker'], self.df_sector['Secteur'])) # ticker -> secteur
        sector_weights_by_date = []

        dates_weights_benchmark = self.df_weights.loc[(self.df_weights.index >= start_date) & 
                                                    (self.df_weights.index <= end_date)].index
        if dates_weights_benchmark.empty:
            raise ValueError("Aucune donnée de poids disponible pour la période spécifiée.")    
        
        # Initialiser les poids de départ
        weights_series = self.df_weights.loc[self.df_weights.index <= all_dates[0]].tail(1).squeeze().fillna(0)
        if len(weights_series)==0:
            weights_series = self.df_weights.iloc[0, :].fillna(0)

        common_tickers = self.get_common_tickers(self.df_prices, weights_series.index)
        filtered_weights = weights_series[weights_series.index.isin(common_tickers) & (weights_series > 0)]
        benchmark_weights = (filtered_weights / filtered_weights.sum()).to_dict()

        for t in tqdm(range(1, len(all_dates)), desc="Running Benchmark"):
            current_date = all_dates[t]

            # Récupérer les rendements du jour
            benchmark_returns = df_returns.loc[df_returns.index == current_date, list(benchmark_weights.keys())].fillna(0).squeeze().to_dict()
            daily_returns_benchmark = np.array([benchmark_returns.get(ticker, 0) for ticker in benchmark_weights.keys()])
            prev_weights_benchmark = np.array(list(benchmark_weights.values()))
            return_bench = np.dot(prev_weights_benchmark, daily_returns_benchmark)

            benchmark_value *= (1 + return_bench)
            stored_benchmark.append(benchmark_value)

            # Mettre à jour les poids si c'est une date de rebalancement
            if current_date in dates_weights_benchmark:
                new_weights = self.df_weights.loc[current_date, :].reindex(self.df_prices.columns).fillna(0).to_dict()
                new_weights = {ticker: weight for ticker, weight in new_weights.items() if weight > 0}
            else:
                # Appliquer le drift
                new_weights = {ticker: benchmark_weights[ticker] * (1 + benchmark_returns[ticker]) for ticker in benchmark_weights}
            benchmark_weights = {ticker: weight / sum(new_weights.values()) for ticker, weight in new_weights.items()}
            sector_weights = {}
            for ticker, weight in benchmark_weights.items():
                sector = sector_map.get(ticker, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

            sector_weights_by_date.append(sector_weights)

        df_benchmark = pd.Series(data=stored_benchmark, index=all_dates, name="Benchmark")
        df_sector_weights = pd.DataFrame(sector_weights_by_date, index=all_dates[1:]) 

        return df_benchmark, df_sector_weights

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
            df_prices (pd.DataFrame): DataFrame contenant les prix avec un index de type datetime.
            frequency (FrequencyType): Fréquence de rebalancement souhaitée.
            custom_freq (str, optional): Fréquence personnalisée (ex: '2M' pour bimensuel, 'Q' pour trimestriel).

        Returns:
            list: Liste des dates de rebalancement.
        """
        if not pd.api.types.is_datetime64_any_dtype(df_prices.index):
            df_prices.index = pd.to_datetime(df_prices.index)

        if frequency == FrequencyType.DAILY:
            # Toutes les dates sont incluses
            return df_prices.index.tolist()

        if frequency == FrequencyType.WEEKLY:
            # Rebalancement hebdomadaire
            return df_prices.groupby(df_prices.index.to_period("W")).apply(lambda group: group.index[-1]).tolist()

        if frequency == FrequencyType.MONTHLY:
            # Rebalancement mensuel
            return df_prices.groupby(df_prices.index.to_period("M")).apply(lambda group: group.index[-1]).tolist()

        if frequency == FrequencyType.QUARTERLY:
            # Rebalancement trimestriel
            return df_prices.groupby(df_prices.index.to_period("Q")).apply(lambda group: group.index[-1]).tolist()
        
        if frequency == FrequencyType.ANNUALLY:
            # Rebalancement trimestriel
            return df_prices.groupby(df_prices.index.to_period("A")).apply(lambda group: group.index[-1]).tolist()
        if custom_freq:
            # Rebalancement avec une fréquence personnalisée (ex: '2M' pour bimensuel)
            return df_prices.groupby(df_prices.index.to_period(custom_freq)).apply(lambda group: group.index[-1]).tolist()

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
        closest_row_universe = self.df_weights[self.df_weights.index <= start_date_dt].tail(1)

        if closest_row_universe.empty:
            raise ValueError("Impossible de récupérer la composition de l'univers avant la start_date.")

        # Récupérer les tickers avec des poids > 0
        tickers_universe = set(
            ticker for ticker in closest_row_universe.loc[:, closest_row_universe.gt(0).iloc[0]].columns.to_list()
        )

        # Si une date de fin est spécifiée, récupérer les tickers actifs dans la plage de dates
        if end_date_dt:
            filtered_universe = self.df_weights[(self.df_weights.index >= start_date_dt) & 
                                                (self.df_weights.index <= end_date_dt)]
            
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
                           n_before: int = 252,
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
        price_dates = price_df.index.sort_values().unique()

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