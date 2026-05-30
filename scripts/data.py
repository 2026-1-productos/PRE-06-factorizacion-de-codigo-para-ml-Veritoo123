import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.25
RANDOM_STATE = 123456
URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_FOLDER = "data/winequality-red/"


def main():

    df = pd.read_csv(URL, sep=";")
    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    os.makedirs(DATA_FOLDER, exist_ok=True)

    x_train.to_csv(os.path.join(DATA_FOLDER, "x_train.csv"))
    y_train.to_csv(os.path.join(DATA_FOLDER, "y_train.csv"))
    x_test.to_csv(os.path.join(DATA_FOLDER, "x_test.csv"))
    y_test.to_csv(os.path.join(DATA_FOLDER, "y_test.csv"))


if __name__ == "__main__":
    main()
