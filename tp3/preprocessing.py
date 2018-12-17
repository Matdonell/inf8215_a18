import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


class TransformationWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, fitation=None, transformation=None):

        self.transformation = transformation
        self.fitation = fitation

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.data_ = None
        self.column_name_ = X.columns[0]
        if self.fitation == None:
            return self

        self.data_ = [self.fitation(X[self.column_name_])]
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)

        if self.data_ != None:
            return pd.DataFrame(X.apply(
                lambda row: pd.Series(self.transformation(row[self.column_name_], self.data_)),
                axis=1
            ))
        else:
            return pd.DataFrame(X.apply(
                lambda row: pd.Series(self.transformation(row[self.column_name_])),
                axis=1
            ))


from sklearn.preprocessing import LabelEncoder


class LabelEncoderP(LabelEncoder):
    def fit(self, X, y=None):
        super(LabelEncoderP, self).fit(X)

    def transform(self, X, y=None):
        return pd.DataFrame(super(LabelEncoderP, self).transform(X))

    def fit_transform(self, X, y=None):
        return super(LabelEncoderP, self).fit(X).transform(X)


# ----------------------------------------------------------------------------------------------
# Code for the preprocessing part
# ----------------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer


def parse_intact(data):
    intact, _ = data.split(" ")
    if intact.lower() == "intact":
        return intact
    elif intact.lower() == "spayed" or intact.lower() == "neutered":
        return "Not intact"
    return "Unknown"


def parse_sex(data):
    _, sex = data.split(" ")
    if sex.lower() == "male" or sex.lower() == "female":
        return sex
    return "Unknown"


def parse_age(data):
    """
        In: The expected data is a string similar to: 2 year
        Out: the corresponding amount of days
    """

    if len(data) > 2 and ((' ' in data) == True):
        n, time_unit = data.split(" ")
        if "year" in time_unit.lower():
            return 365 * n
        elif "month" in time_unit.lower():
            return 30 * n
        elif "week" in time_unit.lower():
            return 7 * n
        elif "day" in time_unit.lower():
            return n
        else:
            return 0
    else:
        return 0


def parse_type(data):
    if data.lower() == "cat" or data.lower() == "dog":
        return data
    return "Unknown"


pipeline_intact = Pipeline([
    ("intact_imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
    ("intact", TransformationWrapper(transformation=parse_intact)),
    ("encode", LabelEncoderP()),
])

pipeline_sex = Pipeline([
    ("sex_imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
    ("sex", TransformationWrapper(transformation=parse_sex)),
    ("encode", LabelEncoderP()),
])

pipeline_age = Pipeline([
    ("age_imputer", SimpleImputer(strategy='constant', fill_value='mean')),
    ("age_converter", TransformationWrapper(transformation=parse_age)),
    ("fillna", SimpleImputer(missing_values=0.0, strategy='mean')),
    ("scaler", StandardScaler()),
])

pipeline_type = Pipeline([
    ("type_imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
    ("encode", LabelEncoderP()),
])

pipeline_sex_upon_outcome = Pipeline([
    ("sex_upon_outcome_imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('sex_upon_outcome_feat', FeatureUnion([
        ('intact', pipeline_intact),
        ('sex', pipeline_sex),
    ])),
])

pipeline_breed = Pipeline([
    ("breed_imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
    ("encode", OneHotEncoder(categories='auto', sparse=False)),
])

full_pipeline = ColumnTransformer([
    ("age", pipeline_age, ["AgeuponOutcome"]),
    ("type", pipeline_type, ["AnimalType"]),
    ("sex", pipeline_sex_upon_outcome, ["SexuponOutcome"]),
    ("breed", pipeline_breed, ["Breed"]),
])

# Load the data
x_train = pd.read_csv("./data/train.csv", header=0)
print(x_train.head())

# Remove the already processed columns
x_train = x_train.drop(columns=["OutcomeSubtype", "AnimalID"])
print(x_train.head())

# Extract specific column and save it in another dataframe
x_train, y_train = x_train.drop(columns=["OutcomeType"]), x_train["OutcomeType"]
print(x_train.head())
print(y_train.head())

# Visualize the data
print(x_train["AnimalType"].value_counts() / len(x_train))
print(x_train["SexuponOutcome"].value_counts() / len(x_train))
print(x_train["Breed"].value_counts() / len(x_train))

# The columns names
column_names = ["AgeuponOutcome", "AnimalType", "SexuponOutcome", "Breed"]

X_train = pd.DataFrame(full_pipeline.fit_transform(x_train))
print(X_train.head())
