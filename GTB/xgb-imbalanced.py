import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
    cross_val_score,
)
from numpy import mean
from sklearn.feature_selection import SelectKBest, chi2, f_classif

import xgboost as xgb


def print_shape(input_):
    print(input_.shape)


def get_x_y(df, target="match"):
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y


# get file
file_path = r"../dataset/processed.csv"
to_drop = ["field_cd"]
raw_data = pd.read_csv(file_path).drop(to_drop, axis=1)


X, y = get_x_y(raw_data)

# select the best 15 features from the dataset
s_f = f_classif
sel = SelectKBest(score_func=s_f, k=15)

cols = X.columns.tolist()
X = sel.fit_transform(X, y)
important_features = [cols[i] for i in sel.get_support(indices=True)]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True)


# model = xgb.XGBClassifier()
model = GradientBoostingClassifier()
model.fit(X_train, y_train)


y_hat = model.predict(X_test)

print(classification_report(y_hat, y_test))
