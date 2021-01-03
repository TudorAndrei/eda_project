from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd


file_path = r"../dataset/processed.csv"
raw_data = pd.read_csv(file_path)


X = raw_data.drop(["goal"], axis=1)
y = raw_data["goal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
)

print(X.shape)
print(len(y))

model = GradientBoostingClassifier()
