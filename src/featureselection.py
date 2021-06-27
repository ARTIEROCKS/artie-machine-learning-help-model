import sys
import yaml
import pandas as pd
import numpy as np


def load(file, columns_to_delete):
    df = pd.read_csv(file, delimiter=',')
    # Deletes the evaluation exercises
    df = df.drop(df[df.exercise_is_evaluation == 1].index)
    # Deletes the columns
    df.drop(columns_to_delete, inplace=True, axis=1)
    return df


# Selects the number of features indicated by the pearson correlation method
def filter_method(cor, target, number_of_features):
    # Gets the correlation with the target
    cor_target = cor[target].abs()

    # Selecting correlations that are numbers
    relevant_features = cor_target[cor_target.isna() == False]
    relevant_features = relevant_features.sort_values(ascending=False)
    return relevant_features[0:number_of_features]


def drop_high_correlated(cor, selected_features, limit):
    to_drop = [column for column in selected_features if any(abs(cor[column]) > limit)]
    return to_drop


params_file = sys.argv[1]
input_csv_file = sys.argv[2]
output_csv_file = sys.argv[3]

with open(params_file, 'r') as fd:
    params = yaml.safe_load(fd)

method = params['selection']['method']
target = params['selection']['target']
number_of_features = params['selection']['number_of_features']
high_correlation = params['selection']['high_correlation']
drop_columns = params['selection']['drop_columns']

df = load(input_csv_file, drop_columns)

if method == 'filter':
    # Using Pearson Correlation
    cor = df.corr(method='pearson')
    cor_tri = cor.abs().where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    selected_features = filter_method(cor_tri, target, number_of_features)
    selected_features_columns = selected_features.to_dict().keys()
    drop_elements = drop_high_correlated(cor_tri, selected_features_columns, 0.90)

    # Gets just the selected elements
    df.drop(columns=[col for col in df if col not in selected_features_columns], inplace=True)

    # Deletes the elements that have been identified to be deleted
    df.drop(drop_elements, inplace=True, axis=1)

    # Gets the x and y from the dataframe
    last_column = len(df.columns) - 1
    X = df.iloc[:, 0:last_column]
    y = df.iloc[:, [last_column]]

# Writes the output file
df.to_csv(output_csv_file, sep=',', index=False)

