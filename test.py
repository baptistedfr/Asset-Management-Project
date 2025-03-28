import pandas as pd
from src import Backtester, FractileMomentumStrategy, Results
from src.tools import FrequencyType
df_price = pd.read_parquet('data/msci_prices.parquet')
df_weight = pd.read_parquet('data/indices.parquet')
df_benchmark = pd.read_parquet('data/MSCI WORLD.parquet')


monthly_strat_percentile = FractileMomentumStrategy(rebalance_frequency = FrequencyType.MONTHLY,lookback_period = 252, nb_fractile = 100)
monthly_strat_5 = FractileMomentumStrategy(rebalance_frequency = FrequencyType.MONTHLY,lookback_period = 252, nb_fractile = 20)

# monthly_strat_quartile = FractileMomentumStrategy(rebalance_frequency = FrequencyType.MONTHLY,lookback_period = 252, nb_fractile = 4)
# quarterly_strat_quartile = FractileMomentumStrategy(rebalance_frequency = FrequencyType.QUARTERLY,lookback_period = 252, nb_fractile = 4)

# monthly_strat_decile = FractileMomentumStrategy(rebalance_frequency = FrequencyType.MONTHLY,lookback_period = 252, nb_fractile = 10)
# quarterly_strat_decile = FractileMomentumStrategy(rebalance_frequency = FrequencyType.QUARTERLY,lookback_period = 252, nb_fractile = 10)

backtest = Backtester(df_price, df_weight, df_benchmark)


result_annualy_percentile = backtest.run(
    "2007-05-01",
    "2024-12-31",
    monthly_strat_percentile,
    fees = 0.0005,
    custom_name = "Percentile Monthly"
)

result_annualy_decile = backtest.run(
    "2007-05-01",
    "2024-12-31",
    monthly_strat_5,
    fees = 0.0005,
    custom_name = "5% Monthly"
)

# result_monthly_quartile = backtest.run(
#     "2007-05-01",
#     "2024-12-31",
#     monthly_strat_quartile,
#     fees = 0.0005,
#     custom_name = "Quartile Monthly"
# )

# result_quarterly_quartile = backtest.run(
#     "2007-05-01",
#     "2024-12-31",
#     quarterly_strat_quartile,
#     fees = 0.0005,
#     custom_name = "Quartile quarterly"
# )

# result_monthly_decile = backtest.run(
#     "2007-05-01",
#     "2024-12-31",
#     monthly_strat_decile,
#     fees = 0.0005,
#     custom_name = "Decile monthly"
# )

# result_quarterly_decile = backtest.run(
#     "2007-05-01",
#     "2024-12-31",
#     quarterly_strat_decile,
#     fees = 0.0005,
#     custom_name = "Decile quarterly"
# )

combined_results = Results.compare_results([result_annualy_percentile, result_annualy_decile])
'''Visualisation des résultats'''
print(combined_results.df_statistics.head(10))
combined_results.ptf_value_plot.show()
combined_results.ptf_drawdown_plot.show()
for plot in combined_results.ptf_weights_plot:
    plot.show()