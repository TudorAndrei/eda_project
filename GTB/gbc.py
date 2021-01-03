import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, chi2, f_classif

file_path = r"../dataset/processed.csv"
to_drop = ['field_cd']
raw_data = pd.read_csv(file_path).drop(to_drop, axis=1)


def print_shape(input_):
    print(input_.shape)


def get_x_y(df, target='match'):
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y


X, y = get_x_y(raw_data)
s_f = chi2
s_f = f_classif
sel = SelectKBest(score_func=s_f, k=15)
X_new = sel.fit_transform(X, y)
imp = sel.get_support(indices=True)

cols = raw_data.columns.tolist()

important_features = [cols[i] for i in imp]

print(important_features)

X = X_new

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True
)


model = GradientBoostingClassifier(random_state=42)

model.fit(X_train, y_train)

y_hat = model.predict(X_test)

print(classification_report(y_hat, y_test))

dataset = raw_data[important_features]

features = pd.Series(model.feature_importances_,
                     index=dataset.columns.values).sort_values(
    ascending=False)

print(features)
