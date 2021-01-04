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


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True)


model = GradientBoostingClassifier(
    n_estimators=100, random_state=42)


imbl_methods = {'eec': EasyEnsembleClassifier(random_state=42,
                                              sampling_strategy=1.,
                                              n_jobs=-1,
                                              base_estimator=model),
                'rub': RUSBoostClassifier(random_state=42,
                                          sampling_strategy=1.,
                                          base_estimator=model)}

for method in imbl_methods.keys():

    imbl = imbl_methods[method]
    imbl.fit(X_train, y_train)
    y_hat_test = imbl.predict(X_test)
    y_hat_train = imbl.predict(X_train)
    print(f"Reults of {method}")
    print(imbl.score(X_test, y_test))
    print("Train data")
    print(classification_report(y_train, y_hat_train))
    print("Test data")
    print(classification_report(y_test, y_hat_test))
