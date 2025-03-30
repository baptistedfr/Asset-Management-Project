import pandas as pd
from src import Backtester, FractileMomentumStrategy, Results
from src.tools import FrequencyType
df_price = pd.read_parquet('data/msci_prices.parquet')
df_weight = pd.read_parquet('data/indices.parquet')
df_benchmark = pd.read_parquet('data/MSCI WORLD.parquet')



def run_strategy(df_price, df_weight, df_benchmark, strategy_class, strategy_params, 
                 rebalance_frequencies, fractiles, start_date, end_date, 
                 output_prefix, fees=0.0005):
    
    backtest = Backtester(df_price, df_weight, df_benchmark)
    results = []

    # Boucle sur les fractiles et les fréquences de rééquilibrage
    for fractile_name, nb_fractile in fractiles.items():
        for rebalance_name, rebalance_type in rebalance_frequencies.items():
            # Initialiser la stratégie avec les paramètres
            strategy = strategy_class(
                rebalance_frequency=rebalance_type,
                nb_fractile=nb_fractile,
                **strategy_params
            )

            # Exécuter le backtest
            result = backtest.run(
                start_date,
                end_date,
                strategy,
                fees=fees,
                custom_name=f"{rebalance_name} {fractile_name}"
            )
            results.append(result)

    # Combiner les résultats
    combined_results = Results.compare_results(results)
    combined_results.df_statistics.to_excel(f"results/{output_prefix}_statistics.xlsx", index=True)
    combined_results.ptf_value_plot.write_image(f"results/{output_prefix}_ptf_value_plot.png")
    combined_results.ptf_drawdown_plot.write_image(f"results/{output_prefix}_ptf_drawdown_plot.png")

    # Afficher les résultats
    print(combined_results.df_statistics.head(20))
    combined_results.ptf_value_plot.show()
    combined_results.ptf_drawdown_plot.show()

    return combined_results

def compute_momentum_1y(df_price, df_weight, df_benchmark):
    fractiles = {"Quartile": 4, "Quintile": 5, "Decile": 10}
    rebalance_frequencies = {"MONTHLY": FrequencyType.MONTHLY, "QUARTERLY": FrequencyType.QUARTERLY}
    strategy_params = {
        "lookback_period": 252,
        "n_ante": 21,
        "mean_reverting": False
    }

    return run_strategy(
        df_price=df_price.copy(),
        df_weight=df_weight.copy(),
        df_benchmark=df_benchmark.copy(),
        strategy_class=FractileMomentumStrategy,
        strategy_params=strategy_params,
        rebalance_frequencies=rebalance_frequencies,
        fractiles=fractiles,
        start_date="2007-05-01",
        end_date="2024-12-31",
        output_prefix="momentum_1y"
    )

def compute_momentum_mean_reverting(df_price, df_weight, df_benchmark):
    fractiles = {"Quartile": 4, "Quintile": 5, "Decile": 10}
    rebalance_frequencies = {"MONTHLY": FrequencyType.MONTHLY, "QUARTERLY": FrequencyType.QUARTERLY}
    strategy_params = {
        "lookback_period": 21,
        "n_ante": 0,
        "mean_reverting": True
    }

    return run_strategy(
        df_price=df_price.copy(),
        df_weight=df_weight.copy(),
        df_benchmark=df_benchmark.copy(),
        strategy_class=FractileMomentumStrategy,
        strategy_params=strategy_params,
        rebalance_frequencies=rebalance_frequencies,
        fractiles=fractiles,
        start_date="2007-05-01",
        end_date="2024-12-31",
        output_prefix="momentum_mean_reverting"
    )

# compute_momentum_1y(df_price, df_weight, df_benchmark)

compute_momentum_mean_reverting(df_price, df_weight, df_benchmark)