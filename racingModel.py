# File: racingModel.py
# Author: Evan Smith
# Purpose: Explore linear regression with NumPy & Pandas using race data to predict likely race winners.

import numpy as np
import pandas as pd

# Log transform to stabilize variance
def log_transform(times):
    return np.log(times)

# Inverse log transform
def inverse_log_transform(log_times):
    return np.exp(log_times)

# Parse race time strings into seconds
def parse_time_to_seconds(time_str):
    try:
        t = str(time_str).strip().lower()

        if t in ['dnf', 'ret', 'lap', '1 lap', '1 la', '+1 lap', '+2 laps', '+3 laps']:
            return np.nan

        if ':' in t and t.count(':') == 2:
            # Format: hours:minutes:seconds, e.g. "1:25:32.100"
            hrs, mins, secs = t.split(':')
            return int(hrs) * 3600 + int(mins) * 60 + float(secs)
        elif ':' in t and t.count(':') == 1:
            # Format: minutes:seconds, e.g. "1:23.456"
            mins, secs = t.split(':')
            return int(mins) * 60 + float(secs)
        elif t.startswith('+'):
            if 'lap' in t:
                return np.nan  # Treat +1 lap, +2 laps etc. as non-comparable
            return float(t[1:-1]) if t.endswith('s') else float(t[1:])
        else:
            return float(t)  # raw seconds
    except Exception:
        return np.nan

# Format float seconds into M:SS.sss (e.g. 84.472 -> 1:24.472)
def format_seconds_to_time_str(seconds):
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:06.3f}"  # e.g. 1:25:32.100
    else:
        return f"{mins}:{secs:06.3f}"              # e.g. 25:32.100


def run():
    print("Loading and preparing mock race data...")

    df = pd.read_csv('data.csv')

    # Parse and convert time strings to seconds
    df['TimeOrGapSeconds'] = df['TotalTimeOrGap'].apply(parse_time_to_seconds)

    # Create a RaceID (important for multi-year expansion)
    df['RaceID'] = df['Year'].astype(str) + "_Race"

    # Determine leader's best time for each race
    df['AbsoluteTimeSeconds'] = np.nan
    for race_id, group in df.groupby('RaceID'):
        leader_time = group[~group['TotalTimeOrGap'].astype(str).str.startswith('+')]['TimeOrGapSeconds'].min()
        df.loc[group.index, 'AbsoluteTimeSeconds'] = group.apply(
            lambda r: leader_time + r['TimeOrGapSeconds']
            if str(r['TotalTimeOrGap']).startswith('+') else r['TimeOrGapSeconds'], axis=1
        )

    # Drop NaNs from DNF or badly formatted data
    df = df.dropna(subset=['AbsoluteTimeSeconds'])

    # Define features and target
    features = ['Year', 'Position', 'Points']
    X = df[features].values.astype(float)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std

    y = df['AbsoluteTimeSeconds'].values.astype(float)
    y_log = log_transform(y)

    # Initialize linear regression weights
    intercept = 0.0
    weights = np.zeros(X_normalized.shape[1])

    # Adam optimizer setup
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    iters = 2500

    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = 0.0
    v_b = 0.0

    print("Training model...")
    for t in range(1, iters + 1):
        preds = intercept + X_normalized.dot(weights)
        error = preds - y_log

        grad_w = (2 / len(y_log)) * X_normalized.T.dot(error)
        grad_b = (2 / len(y_log)) * np.sum(error)

        m_w = beta1 * m_w + (1 - beta1) * grad_w
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
        v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        intercept -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

        if t % 100 == 0 or t == 1:
            mse = np.mean(error ** 2)
            print(f"Iteration {t}: MSE = {mse:.6f}")

    print(f"\nTraining complete.\nIntercept: {intercept:.4f}\nWeights: {weights}")

    # Predictions
    y_log_pred = intercept + X_normalized.dot(weights)
    y_pred = inverse_log_transform(y_log_pred)
    df['PredictedLapTime'] = y_pred

    best_predictions = df.groupby('Driver')['PredictedLapTime'].min().reset_index()
    best_predictions['FormattedTime'] = best_predictions['PredictedLapTime'].apply(format_seconds_to_time_str)

    top5 = best_predictions.sort_values('PredictedLapTime').head(5)

    print("\nTop 5 predicted fastest drivers:")
    print(f"{'Driver':<10} {'LapTime':>10}")
    for _, row in top5.iterrows():
        print(f"{row['Driver']:<10} {row['FormattedTime']:>10}")

if __name__ == '__main__':
    run()
