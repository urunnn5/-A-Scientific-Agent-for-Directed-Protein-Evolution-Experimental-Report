import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'fitness' / 'fitness_mlp.joblib'


def train_model(train):
    encoder = OneHotEncoder(handle_unknown='ignore')
    x = encoder.fit_transform([list(seq) for seq in train['mutated_sequence']])
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64), alpha=0.01, batch_size=256,
        max_iter=80, early_stopping=True, n_iter_no_change=8, random_state=42
    )
    model.fit(x, train['DMS_score'])
    return model, encoder


def evaluate(model, encoder, df):
    true = df['DMS_score'].to_numpy()
    features = encoder.transform([list(seq) for seq in df['mutated_sequence']])
    pred = model.predict(features)
    k = max(1, round(len(df) * 0.05))
    top_true = set(np.argsort(true)[-k:])
    top_pred = np.argsort(pred)[-k:]
    return {
        'MSE': float(mean_squared_error(true, pred)),
        'Pearson': float(pearsonr(true, pred).statistic),
        'Spearman': float(spearmanr(true, pred).statistic),
        'R2': float(r2_score(true, pred)),
        'Top5%_hit_rate': len(top_true & set(top_pred)) / k,
        'Top5%_mean_fitness': float(true[top_pred].mean()),
    }


def report(model, encoder):
    rows = []
    for name in ('train', 'val', 'test'):
        df = pd.read_csv(ROOT / 'data' / f'{name}.csv')
        scores = evaluate(model, encoder, df)
        rows.append({'split': name, **scores})
    result = pd.DataFrame(rows)
    (ROOT / 'results').mkdir(exist_ok=True)
    result.to_csv(ROOT / 'results' / 'mlp_metrics.csv', index=False)
    print(result.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate GFP MLP')
    parser.add_argument('--train', action='store_true', help='重新训练并覆盖现有模型')
    args = parser.parse_args()
    if args.train:
        data = pd.read_csv(ROOT / 'data' / 'train.csv')
        model, encoder = train_model(data)
        joblib.dump((model, encoder), MODEL_PATH)
    else:
        model, encoder = joblib.load(MODEL_PATH)
    report(model, encoder)
