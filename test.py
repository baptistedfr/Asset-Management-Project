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



compute_fractile_momentum_1y(df_price, df_weight, df_benchmark, df_sector)
compute_fractile_momentum_mean_reverting(df_price, df_weight, df_benchmark, df_sector)

compute_idiosyncratic_momentum_1y(df_price, df_weight, df_benchmark, df_sector)
compute_idiosyncratic_momentum_mean_reverting(df_price, df_weight, df_benchmark, df_sector) 


# start_date="2007-05-01"
# end_date="2024-12-31"
# strategy = FractileMomentumStrategy(
#                 rebalance_frequency=FrequencyType.MONTHLY,
#                 nb_fractile=10,
#                 lookback_period=252,
#                 n_ante = 21,
#                 mean_reverting = False,
#                 is_segmentation_sectorial = True,
#                 df_sector = df_sector.copy()
#             )
# backtest = Backtester(df_price, df_weight, df_benchmark, df_sector)
# # Exécuter le backtest
# result = backtest.run(
#     start_date,
#     end_date,
#     strategy,
#     custom_name=f"Idiosyncratic",
#     recompute_benchmark=False
# )

# print(result.df_statistics.head(20))
# result.ptf_value_plot.show()
# result.ptf_drawdown_plot.show()