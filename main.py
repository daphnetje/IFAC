from prepare_income_data import prepare_income_prediction_data
from load_datasets import load_income_data
from IFAC import IFAC
from UBAC import UBAC
from experiments import compare_income_prediction

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #prepare_income_prediction_data()
    compare_income_prediction(coverage=1.0)
    # ifac = IFAC(coverage=0.6, fairness_weight=1.0, sensitive_attributes=['sex', 'race'], reference_group_list=[{'sex': 'Male', 'race': 'White alone'}], val1_ratio=0.1, val2_ratio=0.1, base_classifier='Random Forest')
    # ifac.fit(train_data)
    # ifac.predict(test_data)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
