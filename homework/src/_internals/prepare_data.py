import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.25
RANDOM_STATE = 123456


def prepare_data():
    # descarga de datos
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    # preparacion de datos
    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    # dividir los datos en entrenamiento y testing
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    return x_train, x_test, y_train, y_test
