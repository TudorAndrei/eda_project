import pandas as pd
import seaborn
from imblearn.metrics import classification_report_imbalanced
from numpy import mean
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, make_scorer,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)

from utils import *

rs = 42
feature_selection = False
file_path = "../dataset/processed.csv"
raw_data = pd.read_csv(file_path)


X, y = get_x_y(raw_data)

cols = X.columns.tolist()
index = X.columns.values

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
    X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42)


model = GradientBoostingClassifier(random_state=rs)


model.fit(X_train, y_train)
y_hat = model.predict(X_test)

print(classification_report_imbalanced(y_hat, y_test))
