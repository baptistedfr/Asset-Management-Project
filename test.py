import pandas as pd
from src import Backtester, FractileMomentumStrategy

df_price = pd.read_parquet('data/msci_prices.parquet')
df_indice = pd.read_parquet('data/indices.parquet')

backtest = Backtester(df_price, df_indice)
result = backtest.run(
    "2007-05-01",
    "2024-12-31",
    FractileMomentumStrategy(),
)

print(result.df_statistics)
result.ptf_value_plot.show()
result.ptf_drawdown_plot.show()