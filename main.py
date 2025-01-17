from load_datasets import load_income_data
from IFAC import IFAC

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    income_prediction_data = load_income_data()
    ifac = IFAC(sensitive_attributes=['sex', 'race'], reference_group=[{'sex': 'Male', 'race': 'White alone'}], val1_ratio=0.1, val2_ratio=0.1, base_classifier='Random Forest')
    ifac.fit(income_prediction_data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
