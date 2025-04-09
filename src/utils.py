from src import Backtester, FractileMomentumStrategy, Results, IdiosyncraticMomentumStrategy
from src.tools import FrequencyType
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def run_single_strategy(df_price, df_weight, df_benchmark, df_sector,
                        strategy_class, strategy_params,
                        rebalance_name, rebalance_type,
                        fractile_name, nb_fractile,
                        is_segmentation_sectorial,
                        start_date, end_date, fees):
    
    backtest = Backtester(df_price, df_weight, df_benchmark, df_sector)
    
    strategy = strategy_class(
        rebalance_frequency=rebalance_type,
        nb_fractile=nb_fractile,
        is_segmentation_sectorial=is_segmentation_sectorial,
        df_sector=df_sector.copy(),
        **strategy_params
    )

    result = backtest.run(
        start_date,
        end_date,
        strategy,
        fees=fees,
        custom_name=f"{rebalance_name} {fractile_name}",
        recompute_benchmark=False
    )
    
    return (f"{rebalance_name} {fractile_name}", result)

def run_strategy_multi(df_price, df_weight, df_benchmark, df_sector, 
                 strategy_class, strategy_params, 
                 rebalance_frequencies: dict, fractiles: dict, 
                 is_segmentation_sectorial: bool,
                 start_date, end_date, 
                 output_prefix, fees=0.0):

    tasks = []

    # Préparer toutes les combinaisons
    for fractile_name, nb_fractile in fractiles.items():
        for rebalance_name, rebalance_type in rebalance_frequencies.items():
            tasks.append((rebalance_name, rebalance_type, fractile_name, nb_fractile))

    # Exécuter en parallèle
    results = []
    df_ptf = pd.DataFrame()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                run_single_strategy,
                df_price.copy(), df_weight.copy(), df_benchmark.copy(), df_sector.copy(),
                strategy_class, strategy_params,
                rebalance_name, rebalance_type,
                fractile_name, nb_fractile,
                is_segmentation_sectorial,
                start_date, end_date, fees
            )
            for rebalance_name, rebalance_type, fractile_name, nb_fractile in tasks
        ]

        for future in futures:
            key, result = future.result()
            results.append(result)
            df_ptf[key] = result.ptf_values.copy()
    
    df_ptf["Benchmark"] = result.benchmark_values.copy()

    # Sauvegarde
    output_excel_path = f"results/{output_prefix}.xlsx"
    combined_results = Results.compare_results(results)

    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        combined_results.df_statistics.to_excel(writer, sheet_name="Statistics", index=False)
        df_ptf.to_excel(writer, sheet_name="Portfolio History", index=True)

    combined_results.ptf_value_plot.write_image(f"results/{output_prefix}_ptf_value_plot.png")
    combined_results.ptf_drawdown_plot.write_image(f"results/{output_prefix}_ptf_drawdown_plot.png")

    print(combined_results.df_statistics.head(20))
    combined_results.ptf_value_plot.show()
    combined_results.ptf_drawdown_plot.show()

    return combined_results

def run_strategy(df_price, df_weight, df_benchmark, df_sector, 
                 strategy_class, strategy_params, 
                 rebalance_frequencies : list, fractiles : list, 
                 is_segmentation_sectorial : bool,
                 start_date, end_date, 
                 output_prefix, fees=0.0):
    
    backtest = Backtester(df_price, df_weight, df_benchmark, df_sector)
    results = []
    df_ptf = pd.DataFrame()

    # Boucle sur les fractiles et les fréquences de rééquilibrage
    for fractile_name, nb_fractile in fractiles.items():
        for rebalance_name, rebalance_type in rebalance_frequencies.items():
            # Initialiser la stratégie avec les paramètres
            strategy = strategy_class(
                rebalance_frequency=rebalance_type,
                nb_fractile=nb_fractile,
                is_segmentation_sectorial=is_segmentation_sectorial,
                df_sector = df_sector.copy(),
                **strategy_params
            )

            # Exécuter le backtest
            result = backtest.run(
                start_date,
                end_date,
                strategy,
                fees=fees,
                custom_name=f"{rebalance_name} {fractile_name}",
                recompute_benchmark=False
            )
            results.append(result)

            df_ptf[f"{rebalance_name} {fractile_name}"] = result.ptf_values.copy()
    
    output_excel_path = f"results/{output_prefix}.xlsx"

    # Combiner les résultats
    combined_results = Results.compare_results(results)

    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        combined_results.df_statistics.to_excel(writer, sheet_name="Statistics", index=True)
        df_ptf.to_excel(writer, sheet_name="Portfolio History", index=True)

    combined_results.ptf_value_plot.write_image(f"results/{output_prefix}_ptf_value_plot.png")
    combined_results.ptf_drawdown_plot.write_image(f"results/{output_prefix}_ptf_drawdown_plot.png")

    # Afficher les résultats
    print(combined_results.df_statistics.head(20))
    combined_results.ptf_value_plot.show()
    combined_results.ptf_drawdown_plot.show()

    return combined_results


def compute_fractile_momentum_generic(df_price, df_weight, df_benchmark, df_sector, lookback_period, n_ante, mean_reverting , output_prefix):
    fractiles = {"Quartile": 4, "Quintile": 5, "Decile": 10}
    rebalance_frequencies = {"MONTHLY": FrequencyType.MONTHLY, "QUARTERLY": FrequencyType.QUARTERLY}
    is_segmentation_sectorial_list = [True, False]
    strategy_params = {
        "lookback_period": lookback_period,
        "n_ante": n_ante,
        "mean_reverting": mean_reverting
    }

    results = []

    for is_segmentation_sectorial in is_segmentation_sectorial_list:
        result = run_strategy_multi(
            df_price=df_price.copy(),
            df_weight=df_weight.copy(),
            df_benchmark=df_benchmark.copy(),
            df_sector=df_sector.copy(),
            strategy_class=FractileMomentumStrategy,
            strategy_params=strategy_params,
            rebalance_frequencies=rebalance_frequencies,
            fractiles=fractiles,
            is_segmentation_sectorial=is_segmentation_sectorial,
            start_date="2007-05-01",
            end_date="2024-12-31",
            output_prefix=f"{output_prefix}_{'sectorial' if is_segmentation_sectorial else 'non_sectorial'}"
        )
        results.append(result)

    return results


def compute_fractile_momentum_1y(df_price, df_weight, df_benchmark, df_sector):
    return compute_fractile_momentum_generic(
        df_price, df_weight, df_benchmark, df_sector,
        lookback_period=252, n_ante=21, mean_reverting=False,
        output_prefix="fractile_momentum_1y"
    )

def compute_fractile_momentum_mean_reverting(df_price, df_weight, df_benchmark, df_sector):
    return compute_fractile_momentum_generic(
        df_price, df_weight, df_benchmark, df_sector,
        lookback_period=21, n_ante=0, mean_reverting=True,
        output_prefix="fractile_momentum_mean_reverting"
    )


def compute_idiosyncratic_momentum_generic(df_price, df_weight, df_benchmark, df_sector, lookback_period, n_ante, mean_reverting, output_prefix):
    fractiles = {"Quartile": 4, "Quintile": 5, "Decile": 10}
    rebalance_frequencies = {"MONTHLY": FrequencyType.MONTHLY, "QUARTERLY": FrequencyType.QUARTERLY}
    is_segmentation_sectorial_list = [True, False]
    strategy_params = {
        "lookback_period": lookback_period,
        "n_ante": n_ante,
        "mean_reverting": mean_reverting
    }

    results = []

    for is_segmentation_sectorial in is_segmentation_sectorial_list:
        result = run_strategy_multi(
            df_price=df_price.copy(),
            df_weight=df_weight.copy(),
            df_benchmark=df_benchmark.copy(),
            df_sector=df_sector.copy(),
            strategy_class=IdiosyncraticMomentumStrategy,
            strategy_params=strategy_params,
            rebalance_frequencies=rebalance_frequencies,
            fractiles=fractiles,
            is_segmentation_sectorial=is_segmentation_sectorial,
            start_date="2007-05-01",
            end_date="2024-12-31",
            output_prefix=f"{output_prefix}_{'sectorial' if is_segmentation_sectorial else 'non_sectorial'}"
        )
        results.append(result)

    return results

def compute_idiosyncratic_momentum_1y(df_price, df_weight, df_benchmark, df_sector):
    return compute_idiosyncratic_momentum_generic(
        df_price, df_weight, df_benchmark, df_sector,
        lookback_period=252, n_ante=21, mean_reverting=False,
        output_prefix="idiosyncratic_momentum_1y"
    )


def compute_idiosyncratic_momentum_mean_reverting(df_price, df_weight, df_benchmark, df_sector):
    return compute_idiosyncratic_momentum_generic(
        df_price, df_weight, df_benchmark, df_sector,
        lookback_period=21, n_ante=0, mean_reverting=True,
        output_prefix="idiosyncratic_momentum_mean_reverting"
    )

