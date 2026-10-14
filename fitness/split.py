from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def split_data(seed=42):
    df = pd.read_csv(ROOT / 'data' / 'GFP_AEQVI_Sarkisyan_2016.csv')
    train, rest = train_test_split(df, test_size=0.30, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed)
    for name, data in (('train', train), ('val', val), ('test', test)):
        data.to_csv(ROOT / 'data' / f'{name}.csv', index=False)


if __name__ == '__main__':
    split_data()
