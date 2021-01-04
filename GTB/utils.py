from sklearn.metrics import classification_report
import pandas as pd


def print_shape(input_):
    print(input_.shape)


def get_x_y(df, target="match"):
    X = df.drop([target], axis=1)
    y = df[target]
    return X, y


def print_classification_report(type, model, X, y):
    y_hat = model.predict(X)

    print(f"{type} classification")
    print(classification_report(y_hat, y))


def print_important_feature(model, index):
    features = pd.Series(model.feature_importances_,
                         index=index).sort_values(
        ascending=False
    )

    print(features)
