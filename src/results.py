from dataclasses import dataclass
from typing import Union
import plotly.graph_objects as go
from typing import Optional
import pandas as pd
import numpy as np
from functools import cached_property
from .tools import FrequencyType
from datetime import datetime

@dataclass
class Results:

    def __init__(self, 
             ptf_values: pd.Series, 
             strategy_name: str, 
             data_frequency: FrequencyType, 
             ptf_weights: pd.DataFrame = None,
             df_statistics: pd.DataFrame = None, 
             ptf_value_plot: go.Figure = None,  
             ptf_drawdown_plot: go.Figure = None, 
             ptf_weights_plot: Union[go.Figure, list[go.Figure]] = None,
             benchmark_values : pd.Series = None):
        """
        Initialise une instance de la classe Results.

        Args:
            ptf_values (pd.Series): Valeurs du portefeuille au fil du temps (indexé par date).
            strategy_name (str): Nom de la stratégie utilisée.
            data_frequency (FrequencyType): Fréquence des données (ex: DAILY, MONTHLY).
            ptf_weights (pd.DataFrame, optional): Poids des actifs dans le portefeuille au fil du temps. Default: None.
            df_statistics (pd.DataFrame, optional): DataFrame contenant les statistiques calculées pour la stratégie. Default: None.
            ptf_value_plot (go.Figure, optional): Graphique de l'évolution des valeurs du portefeuille. Default: None.
            ptf_drawdown_plot (go.Figure, optional): Graphique des drawdowns du portefeuille. Default: None.
            ptf_weights_plot (Union[go.Figure, list[go.Figure]], optional): Graphique(s) des poids des actifs. Default: None.
        """
        self.ptf_values: pd.Series = ptf_values
        self.strategy_name: str = strategy_name
        self.data_frequency: FrequencyType = data_frequency
        self.ptf_weights: Optional[pd.DataFrame] = ptf_weights
        self.df_statistics: Optional[pd.DataFrame] = df_statistics
        self.ptf_value_plot: Optional[go.Figure] = ptf_value_plot
        self.ptf_drawdown_plot: Optional[go.Figure] = ptf_drawdown_plot
        self.ptf_weights_plot: Optional[Union[go.Figure, list[go.Figure]]] = ptf_weights_plot
        self.benchmark_values : Optional[pd.DataFrame] = benchmark_values
        self.rf = 0.0
    """---------------------------------------------------------------------------------------
    -                                 Generate Statistics                                    -
    ---------------------------------------------------------------------------------------"""

    @cached_property
    def ptf_returns(self) -> list[float]:
        return list(pd.Series(self.ptf_values).pct_change().iloc[1:])
    
    @cached_property
    def benchmark_returns(self) -> list[float]:
        if self.benchmark_values is not None:
            return list(pd.Series(self.benchmark_values).pct_change().iloc[1:])
        return None
    
    @property
    def total_return(self) -> float:
        return (self.ptf_values.iloc[-1] / self.ptf_values.iloc[0]) - 1
        
    @property
    def annualized_return(self) -> float:
        return (self.ptf_values.iloc[-1]/self.ptf_values.iloc[0])**(self.data_frequency.value/len(self.ptf_values)) - 1
    
    @property
    def annualized_benchmark_return(self) -> float:
        if self.benchmark_values is not None:
            return (self.benchmark_values.iloc[-1]/self.benchmark_values.iloc[0])**(self.data_frequency.value/len(self.benchmark_values)) - 1
        return None

    @cached_property
    def annualized_vol(self) -> float:
        return np.std(self.ptf_returns) * np.sqrt(self.data_frequency.value)
    
    @cached_property
    def annualized_downside_vol(self) -> float:
        downside_returns = [r for r in self.ptf_returns if r < 0]
        downside_std = np.std(downside_returns, ddof=1) * np.sqrt(self.data_frequency.value)
        return downside_std

    @property
    def sharpe_ratio(self) -> float:
        return (self.annualized_return-self.rf)/self.annualized_vol
    
    @property
    def sortino_ratio(self) -> float:
        return (self.annualized_return-self.rf) / self.annualized_downside_vol
    
    @property
    def drawdowns(self)-> float:
        portfolio_values = np.array(self.ptf_values)
        previous_peaks = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - previous_peaks) / previous_peaks
        return drawdowns

    @property
    def max_drawdown(self) -> float:
        max_drawdown = np.min(self.drawdowns)
        return max_drawdown

    @property
    def max_drawdown_date(self) -> pd.Timestamp:
        """Date du drawdown maximal."""
        min_index = np.argmin(self.drawdowns)
        return self.ptf_values.index[min_index]
    
    @property
    def calmar_ratio(self):
        """Calcul du ratio de Calmar."""
        return (self.annualized_return) / abs(self.max_drawdown) if self.max_drawdown !=0 else np.nan
    
    def compute_VaR(self, alpha: float = 0.95) -> float:
        """
        Compute the Value at Risk (VaR) of the strategy at the given confidence level (alpha).

        Args:
            alpha (float): Confidence level for the VaR calculation (e.g., 0.95 for 95% confidence).

        Returns:
            float: The computed Value at Risk (VaR).

        Raises:
            ValueError: If alpha is not in the range (0, 1).
            ValueError: If the portfolio returns are empty.
        """
        # Validate inputs
        if not (0 < alpha < 1):
            raise ValueError("Alpha must be a float between 0 and 1.")
        if len(self.ptf_returns) == 0:
            raise ValueError("Portfolio returns cannot be empty.")
        # Compute VaR
        var = np.percentile(np.array(self.ptf_returns), (1 - alpha) * 100)
        return var
    
    def compute_CVaR(self, alpha: float = 0.95) -> float:
        """
        Compute the Conditional Value at Risk (CVaR) of the strategy at the given confidence level (alpha).

        Args:
            alpha (float): Confidence level for the CVaR calculation (e.g., 0.95 for 95% confidence).

        Returns:
            float: The computed Conditional Value at Risk (CVaR).
        """
        var_threshold = self.compute_VaR(alpha)
        # Compute CVaR as the average of returns below the VaR threshold
        cvar = np.mean([r for r in self.ptf_returns if r <= var_threshold])
        return cvar
    
    @property
    def beta(self) -> float:
        if self.benchmark_values is not None:
            covariance = np.cov(self.ptf_returns, self.benchmark_returns)[0, 1]
            benchmark_variance = np.var(self.benchmark_returns)
            return covariance / benchmark_variance
        return None
    
    @property
    def treynor_ratio(self) -> float:
        if self.benchmark_values is not None:
            return (self.annualized_return - self.rf)/self.beta
        return None
    
    @property
    def tracking_error(self) ->float : 
        if self.benchmark_values is not None:
            active_returns = np.array(self.ptf_returns) - np.array(self.benchmark_returns)
            return np.std(active_returns) * np.sqrt(self.data_frequency.value)
        return None
    
    @property
    def alpha(self) -> float:
        if self.benchmark_values is not None:
            return self.annualized_return - self.rf - self.beta*(self.annualized_benchmark_return - self.rf)
        return None
    
    @property
    def information_ratio(self) -> float:
        if self.benchmark_values is not None:
            return self.alpha/self.tracking_error
        return None


    def get_statistics(self) -> pd.DataFrame:
        """
        Compute the basic statistics of the strategy
        """
        metrics = [
            self.total_return,
            self.annualized_return, 
            self.annualized_vol, 
            self.annualized_downside_vol,
            self.sharpe_ratio, 
            self.sortino_ratio, 
            self.max_drawdown,
            self.max_drawdown_date,
            self.calmar_ratio,
            self.compute_VaR(),
            self.compute_CVaR(),
            self.beta,
            self.treynor_ratio,
            self.tracking_error,
            self.alpha,
            self.information_ratio,
        ]

        data = {
            "Metrics": ["Total Return", "Annualized Return", "Volatility", "Downside Volatility",
                        "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Max Drawdown Date", "Calmar Ratio",
                        "VaR 95%", "CVaR 95%", 
                        "Beta", "Treynor Ratio", "Tracking Error", "Alpha", "Information Ratio"],
            self.strategy_name: metrics
        }
        formatted_data = []

        for i, metric in enumerate(data[self.strategy_name]):
            if pd.isnull(metric):  # Si une métrique est NaN
                formatted_data.append("N/A")
            elif i in [0, 1, 2, 3, 6, 9, 10, 12, 13, 14]:  # Total Return, Annualized Return, Volatility, Max Drawdown (en pourcentage)
                formatted_data.append("{:.2%}".format(metric))
            elif i == 7:
                formatted_data.append(datetime.strftime(metric, "%Y-%m-%d"))
            else:  # Ratios (Sharpe, Sortino)
                formatted_data.append("{:.2f}".format(metric))
        data[self.strategy_name] = formatted_data
        df = pd.DataFrame(data)
        self.df_statistics = df

    """---------------------------------------------------------------------------------------
    -                                   Generate Plots                                       -
    ---------------------------------------------------------------------------------------"""

    def create_plots(self) :
        self.strat_plot()
        self.drawdown_plot()
        self.weights_plot()
        
    def strat_plot(self) :

        strat_values = list(self.ptf_values)
        dates = pd.to_datetime(self.ptf_values.index)
        
        if self.strategy_name == "Benchmark":
            fig = go.Figure(data=go.Scatter(x=dates, y=strat_values,name=self.strategy_name, line=dict(dash='dot')))
        else : 
            fig = go.Figure(data=go.Scatter(x=dates, y=strat_values,name=self.strategy_name))

        fig.update_layout(title=f'Strategy performance {self.strategy_name}', 
                          xaxis_title='Dates', 
                          yaxis_title='Portfolio Values', 
                          font=dict(family="Courier New, monospace", size=14,color="RebeccaPurple"))
        
        self.ptf_value_plot = fig
    
    def drawdown_plot(self):
        drawdown_values = list(self.drawdowns*100)
        dates = pd.to_datetime(self.ptf_values.index)
        
        if self.strategy_name == "Benchmark":
            fig = go.Figure(data=go.Scatter(x=dates, y=drawdown_values,name=self.strategy_name, line=dict(dash='dot')))
        else : 
            fig = go.Figure(data=go.Scatter(x=dates, y=drawdown_values,name=self.strategy_name))

        fig.update_layout(title=f'Drawdowns of the strategy {self.strategy_name}', 
                          xaxis_title='Dates', 
                          yaxis_title='Drawdown Values (%)', 
                          font=dict(family="Courier New, monospace", size=14,color="RebeccaPurple"))
        
        self.ptf_drawdown_plot = fig
    
    def weights_plot(self):
        if self.ptf_weights is not None :
            min_weight = min(self.ptf_weights.min().min(),0)
            max_weight = 1

            fig = go.Figure()
            '''Attribut "stackgroup" permet d'empiler les plots'''
            for column in self.ptf_weights.columns:
                fig.add_trace(go.Scatter(x=self.ptf_weights.index, y=self.ptf_weights[column],
                                         mode='lines', stackgroup='one', name=column))

            fig.update_layout(title=f"Evolution of portfolio weights for strategy : {self.strategy_name}", xaxis_title="Date",
                yaxis_title="Weight (%)", yaxis=dict(tickformat=".0%", range=[min_weight, max_weight]),
                legend_title="Assets", hovermode="x unified")
            
            self.ptf_weights_plot = fig

    """---------------------------------------------------------------------------------------
    -                                 Results Comparison                                     -
    ---------------------------------------------------------------------------------------"""

    @staticmethod
    def compare_results(results : list["Results"]) -> "Results":
        """
        Compare multiple strategy results and returns a new Results instance.
        If the strategy is compared with the benchmark, the comparaison result is named after the strategy to anable comparaison with other strategies afterward.
        
        Args:
            results (list[Results]) = list of different strategy results
        
        Returns:
            Results: A new Results object containing the combined statistics and comparison plot.
        """

        combined_statistics = Results.combine_df(results)
        value_plot = Results.combine_value_plot(results)
        drawdowns_plots = Results.combine_drawdown_plots(results)
        weight_plots = Results.combine_weight_plot(results)

        if "Benchmark" in [res.strategy_name for res in results]:
            strat_name = [res.strategy_name for res in results if res.strategy_name != "Benchmark"][0]
        else :
            strat_name = "Comparaison"

        return Results(ptf_values=results[0].ptf_values, ptf_weights=results[0].ptf_weights,
                       df_statistics=combined_statistics, ptf_value_plot=value_plot, ptf_drawdown_plot=drawdowns_plots,
                       ptf_weights_plot=weight_plots, strategy_name=strat_name,
                       data_frequency=results[0].data_frequency)
    
    @staticmethod
    def combine_df(results : list["Results"]) -> go.Figure:
        """
        Combine the statistics dataframes and filter the columns to avoid having the benchmark twice

        Args:
            results (list[Results]) : list of different strategy results

        Returns:
            go.Figure : combined plots of strategy values over time
        """

        '''Combine the statistics DataFrames'''
        combined_statistics = pd.DataFrame(columns=["Metrics"])
        is_benchmark= False
        metrics_order = []
        for result in results:
            if result.df_statistics is None:
                result.get_statistics()
            df_stats = result.df_statistics.copy()
            if not metrics_order:
                metrics_order = df_stats["Metrics"].tolist()
            '''Get only one backtest column'''
            if "Benchmark" in df_stats.columns and is_benchmark is False:
                is_benchmark = True
            elif "Benchmark" in df_stats.columns and is_benchmark is True:
                df_stats = df_stats.drop('Benchmark', axis=1)

            combined_statistics = pd.merge(
                combined_statistics, df_stats, on="Metrics", how="outer"
            )
        '''Reorganise the columns'''
        cols = [col for col in combined_statistics.columns if col != "Benchmark"]
        if is_benchmark:
            cols+=["Benchmark"]
        combined_statistics = combined_statistics[cols]
        combined_statistics["Metrics"] = pd.Categorical(
            combined_statistics["Metrics"], categories=metrics_order, ordered=True
        )
        combined_statistics = combined_statistics.sort_values("Metrics").reset_index(drop=True)

        return combined_statistics
    
    @staticmethod
    def combine_value_plot(results : list["Results"]) -> go.Figure:
        """
        Combine the value plots by adding each trace or each plot results in a new plotly figure.
        To avoid adding twice the benchmark plot we check if the benchmark as already been added in the fig by check the scatter names.

        Args:
            results (list[Results]) : list of different strategy results

        Returns:
            go.Figure : combined plots of strategy values over time
        """
        fig = go.Figure()
        existing_names = set()
        for result in results:
            if result.ptf_value_plot is None:
                result.create_plots()
            for scatter in result.ptf_value_plot.data:
                if scatter.name not in existing_names:
                    fig.add_trace(scatter)
                    existing_names.add(scatter.name)

        fig.update_layout(title="Multiple Strategies Performance Comparison", xaxis_title="Date",
            yaxis_title="Portfolio Value", font=dict(family="Courier New, monospace", size=14, color="RebeccaPurple"))
        
        return fig
    
    @staticmethod
    def combine_drawdown_plots(results : list["Results"]) -> go.Figure:
        """
        Combine the drawdown plots by adding each trace or each plot results in a new plotly figure.
        To avoid adding twice the benchmark plot we check if the benchmark as already been added in the fig by check the scatter names.

        Args:
            results (list[Results]) : list of different strategy results

        Returns:
            go.Figure : combined plots of drawdowns values over time
        """
        fig = go.Figure()
        existing_names = set()
        for result in results:
            for scatter in result.ptf_drawdown_plot.data:
                if scatter.name not in existing_names:
                    fig.add_trace(scatter)
                    existing_names.add(scatter.name)

        fig.update_layout(title="Multiple Strategies Drawdown Comparison", xaxis_title="Date",
            yaxis_title="Drawdown Values (%)", font=dict(family="Courier New, monospace", size=14, color="RebeccaPurple"))
        
        return fig
    
    @staticmethod
    def combine_weight_plot(results : list["Results"]) -> go.Figure:
        """
        Combine the weights plots.
        As the function is also used to compare with the benchmark, we want to return only one weight plot if it's the case else we return both.

        Args:
            results (list[Results]) : list of different strategy results

        Returns:
            go.Figure : combined plots of strategy weights over time
        """

        if "Benchmark" not in [res.strategy_name for res in results]:
            '''In this case, we can store all weights evolution in a list of figures'''
            weight_plots = []
            for result in results:
                if result.ptf_weights_plot is not None:
                    weight_plots.append(result.ptf_weights_plot)
        else :
            '''Return only the weigt plot of the strategy and not the benchmark'''
            strat_name = [res.strategy_name for res in results if res.strategy_name != "Benchmark"][0]
            weight_plots = [res.ptf_weights_plot for res in results if res.strategy_name == strat_name][0]

        return weight_plots