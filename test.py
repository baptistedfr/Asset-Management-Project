import pandas as pd
from src import Backtester, FractileMomentumStrategy, Results
from src.tools import FrequencyType
df_price = pd.read_parquet('data/msci_prices.parquet')
df_weight = pd.read_parquet('data/indices.parquet')
df_benchmark = pd.read_parquet('data/MSCI WORLD.parquet')

list_fractile = {"Quartile" : 4, "Quintile":5, "Decile":10}
list_rebalance = {"MONTHLY":FrequencyType.MONTHLY, "QUARTERLY":FrequencyType.QUARTERLY}

backtest = Backtester(df_price, df_weight, df_benchmark)
results = []
for type, nb_fractile in list_fractile.items():
    for name_rebalance, rebalance_type in list_rebalance.items():
        strategy = FractileMomentumStrategy(rebalance_frequency = rebalance_type,
                                            lookback_period = 252, 
                                            nb_fractile = nb_fractile,
                                            n_ante=0)

        result = backtest.run(
            "2007-05-01",
            "2024-12-31",
            strategy,
            fees = 0.0005,
            custom_name = f"{name_rebalance} {type}"
        )
        results.append(result)


combined_results = Results.compare_results(results)
combined_results.df_statistics.to_excel("results/statistics/no_ante.xlsx", index = True)
'''Visualisation des résultats'''
print(combined_results.df_statistics.head(20))
combined_results.ptf_value_plot.show()
combined_results.ptf_drawdown_plot.show()