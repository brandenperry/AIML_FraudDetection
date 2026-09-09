from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def load_data(path=DATA_PATH, test_size=0.2, random_state=42):
    df = pd.read_csv(path)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df.drop(columns=["Class"]))
    df_scaled = pd.DataFrame(scaled_values, columns=df.columns[:-1])
    df_scaled["Class"] = df["Class"]

    X = df_scaled.drop("Class", axis=1)
    y = df_scaled["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Unscaled dollar amounts for the test set, for cost-based evaluation
    amount_test = df.loc[X_test.index, "Amount"]

    return X_train, X_test, y_train, y_test, amount_test
