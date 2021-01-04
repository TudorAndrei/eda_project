from collections import Counter

import pandas as pd
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from utils import *
import xgboost as xgb

feature_selection = False
file_path = "../dataset/processed.csv"
raw_data = pd.read_csv(file_path)
# raw_data.drop(to_drop, axis=1, inplace=True)


X, y = get_x_y(raw_data)
# save the columns of the dataframe
cols = X.columns.tolist()

index = X.columns.values

print(Counter(y))


if feature_selection == True:
    s_f = f_classif
    sel = SelectKBest(score_func=s_f, k=15)
    X_new = sel.fit_transform(X, y)
    imp = sel.get_support(indices=True)
    important_features = [cols[i] for i in imp]
    index = raw_data[important_features].columns.values

    print(f"The important features are : \n\t{important_features}")
    X = X_new


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y)


models = {'xgb': xgb.XGBClassifier(use_label_encoder=False, verbosity=0,
                                   n_jobs=-1),
          'sklearn-gbc': GradientBoostingClassifier()}

for key in models.keys():
    print(key)
    estimator = models[key]
    eec = EasyEnsembleClassifier(
        random_state=42, sampling_strategy=0.5, base_estimator=estimator)

    eec.fit(X_train, y_train)

    y_hat = eec.predict(X_test)
    y_hat_train = eec.predict(X_train)

    print("Training classification")
    print(classification_report_imbalanced(y_hat_train, y_train))
    print("Testing classification")
    print(classification_report_imbalanced(y_hat, y_test))


features = pd.Series(model.feature_importances_, index=index).sort_values(
    ascending=False
)

print(features)
