import pandas as pd
from src import Backtester, FractileMomentumStrategy, Results, IdiosyncraticMomentumStrategy
from src.tools import FrequencyType
from src.utils import (compute_idiosyncratic_momentum_1y, 
                       compute_idiosyncratic_momentum_mean_reverting,
                       compute_fractile_momentum_1y, 
                       compute_fractile_momentum_mean_reverting)

df_price = pd.read_parquet('data/msci_prices.parquet')
df_weight = pd.read_parquet('data/indices.parquet')
df_benchmark = pd.read_parquet('data/MSCI WORLD.parquet')
df_sector = pd.read_parquet('data/sectors.parquet')



# compute_fractile_momentum_1y(df_price, df_weight, df_benchmark, df_sector)
# compute_fractile_momentum_mean_reverting(df_price, df_weight, df_benchmark, df_sector)

# compute_idiosyncratic_momentum_1y(df_price, df_weight, df_benchmark, df_sector)
# compute_idiosyncratic_momentum_mean_reverting(df_price, df_weight, df_benchmark, df_sector) 


start_date="2007-05-01"
end_date="2024-12-31"
strategy_monthly_quintile = FractileMomentumStrategy(
                rebalance_frequency=FrequencyType.MONTHLY,
                nb_fractile=4,
                lookback_period=21,
                n_ante = 0,
                mean_reverting = True,
                is_segmentation_sectorial = True,
                df_sector = df_sector.copy(),
            )

backtest = Backtester(df_price, df_weight, df_benchmark, df_sector)
# Exécuter le backtest
result = backtest.run(
    start_date,
    end_date,
    strategy_monthly_quintile,
    custom_name=f"Mean Reverting MONTHLY Quartile Sectoriel",
    recompute_benchmark=False,
    fees = 0.0
)

strategy_monthly_decile = IdiosyncraticMomentumStrategy(
                rebalance_frequency=FrequencyType.QUARTERLY,
                nb_fractile=4,
                lookback_period=21,
                n_ante = 0,
                mean_reverting = True,
                is_segmentation_sectorial = True,
                df_sector = df_sector.copy(),
            )

# Exécuter le backtest
result2 = backtest.run(
    start_date,
    end_date,
    strategy_monthly_decile,
    custom_name=f"Idiosyncratic Mean Reverting QUARTERLY Quartile Sectoriel",
    recompute_benchmark=False,
    fees = 0.0
)

global_results = Results.compare_results([result, result2], custom_name="Comparaison des stratégies mean reverting")
print(global_results.df_statistics.head(20))
global_results.ptf_value_plot.show()
global_results.ptf_drawdown_plot.show()