import matplotlib.pyplot as plt
import pandas as pd
from imblearn.metrics import classification_report_imbalanced
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, plot_confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

from utils import *

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


model = GradientBoostingClassifier(random_state=42)

model.fit(X_train, y_train)
y_hat = model.predict(X_test)
y_hat_train = model.predict(X_train)

print(classification_report_imbalanced(y_test, y_hat))
print(classification_report_imbalanced(y_train, y_hat_train))

disp = plot_confusion_matrix(model, X_test, y_test,
                             cmap=plt.cm.Blues)
disp.ax_.set_title('title')

plt.show()
