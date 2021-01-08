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


X_train = pd.read_csv("../dataset/train.csv")
X_test = pd.read_csv("../dataset/test.csv")
y_train = X_train.match
y_test = X_test.match
X_train.drop('match', axis=1)
X_test.drop('match', axis=1)


model = GradientBoostingClassifier(random_state=42)

model.fit(X_train, y_train)

y_hat = model.predict(X_test)
print(accuracy_score(y_test, y_hat))

disp = plot_confusion_matrix(model, X_test, y_test,
                             cmap=plt.cm.Blues)
disp.ax_.set_title('title')

plt.show()
