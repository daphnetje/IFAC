from Dataset import Dataset
import pandas as pd

def load_income_data():
    raw_data = pd.read_csv('income_sample.csv')
    descriptive_dataframe = raw_data[
        ['age', 'marital status', 'education', 'workinghours', 'workclass', 'occupation', 'race', 'sex', 'income']]
    numerical_dataframe = raw_data[
        ['age_num', 'marital status', 'education_num', 'workinghours_num', 'workclass', 'occupation', 'race', 'sex',
         'income']]

    numerical_dataframe = numerical_dataframe.astype(
        {'education_num': 'int32', 'workinghours_num': 'int32', 'age_num': 'int32'})

    print(descriptive_dataframe['race'].unique())
    categorical_features = ['marital status', 'occupation', 'workclass', 'race', 'sex']
    dataset = Dataset(descriptive_dataframe, numerical_dataframe, decision_attribute="income", undesirable_label="low",
                      desirable_label="high", categorical_features=categorical_features,
                      distance_function=distance_function_income_pred)

    return dataset

#order of features: ['age_num', 'marital status', 'education_num', 'workinghours_num', 'workclass', 'occupation', 'race', 'sex', 'income']]
def distance_function_income_pred(x1, x2):
    age_diff = abs(x1[0] - x2[0]) / 6

    if x1[1] == x2[1]:
        marital_status_diff = 0
    else:
        marital_status_diff = 0.5

    education_diff = abs(x1[2] - x2[2]) / 9

    workinghours_diff = abs(x1[3] - x2[3])/3

    if x1[4] == x2[4]:
        workclass_diff = 0
    else:
        workclass_diff = 0.5

    if x1[5] == x2[5]:
        occupation_diff = 0
    else:
        occupation_diff = 0.5

    return age_diff + marital_status_diff + education_diff + workinghours_diff + workclass_diff + occupation_diff
