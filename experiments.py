from load_datasets import load_income_data
from IFAC import IFAC
from IFAC.PD_itemset import generate_potentially_discriminated_itemsets
from UBAC import UBAC
from performance_measuring import extract_performance_df_over_non_rejected_instances
import pandas as pd
import numpy as np
from visualizations import visualize_averaged_performance_measure_for_single_and_intersectional_axis

def compare_income_prediction(coverage):
    income_prediction_data = load_income_data()

    train_data, test_data_array = income_prediction_data.split_into_train_and_multiple_test_sets(train_size=12000, number_of_test_sets=4)
    pd_itemsets = generate_potentially_discriminated_itemsets(train_data, ['sex', 'race'])

    ubac = UBAC(coverage=coverage, val_ratio=0.2, base_classifier='Random Forest')
    ubac.fit(train_data)

    ifac = IFAC(coverage=coverage, fairness_weight=1.0, sensitive_attributes=['sex', 'race'], reference_group_list=[{'sex': 'Male', 'race': 'White alone'}], val1_ratio=0.2, val2_ratio=0.2, base_classifier='Random Forest')
    ifac.fit(train_data)

    all_performances = pd.DataFrame([])
    iteration = 1
    for test_data in test_data_array:
        print("Iteration: ", iteration)
        ubac_predictions = ubac.predict(test_data)
        ubac_performance = extract_performance_df_over_non_rejected_instances(classification_method="UBAC", data=test_data, predictions=ubac_predictions, pd_itemsets=pd_itemsets)

        ifac_predictions, ifac_flips = ifac.predict(test_data)
        print(len(ifac_flips))
        print(ifac_flips)
        ifac_performance = extract_performance_df_over_non_rejected_instances(classification_method="IFAC", data=test_data, predictions=ifac_predictions, pd_itemsets=pd_itemsets)

        all_performances = pd.concat([all_performances, ubac_performance, ifac_performance])
        iteration += 1
    averaged_performances = average_performance_results_over_multiple_splits(all_performances)


    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances,
                                                                              "Positive Dec. Ratio")
    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FPR")
    visualize_averaged_performance_measure_for_single_and_intersectional_axis(averaged_performances, "FNR")


def average_performance_results_over_multiple_splits(performance_dataframes):
    performance_measures_of_interest = ['Accuracy', 'FPR', 'FNR', 'Positive Dec. Ratio', 'Number of instances']
    summary_df = performance_dataframes.groupby(['Classification Type', 'Group', "Sensitive Features"])[
        "Group", "Sensitive Features", "Accuracy", "FPR", "FNR", "Positive Dec. Ratio", "Number of instances"].agg(
        {'Accuracy': ['mean', 'std'], 'FPR': ['mean', 'std'], 'FNR': ['mean', 'std'], 'Positive Dec. Ratio': ['mean', 'std'], 'Number of instances': ['mean', 'std']})
    summary_df.columns = [' '.join(col).strip() for col in summary_df.columns.values]
    summary_df.reset_index(inplace=True)

    # calculate upper and lower bounds of confidence intervals
    for performance_measure in performance_measures_of_interest:
        summary_df[performance_measure + ' ci'] = 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(performance_dataframes)))
        summary_df[performance_measure + ' ci_low'] = summary_df[performance_measure + ' mean'] - 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(performance_dataframes)))
        summary_df[performance_measure + ' ci_high'] = summary_df[performance_measure + ' mean'] + 1.96 * (
                    summary_df[performance_measure + ' std'] / np.sqrt(len(performance_dataframes)))

        if performance_measure != "Number of instances":
            # make sure confidence intervals range from 0 to 1
            summary_df[performance_measure + ' ci_low'] = summary_df[
                performance_measure + ' ci_low'].apply(lambda x: 0 if x < 0 else x)
            summary_df[performance_measure + ' ci_high'] = summary_df[
                performance_measure + ' ci_high'].apply(lambda x: 1 if x > 1 else x)

    return summary_df