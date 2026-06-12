from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor

os.environ.setdefault("KERAS_BACKEND", "torch")

from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, LSTM, SimpleRNN
from keras.optimizers import Adam


MODEL_TYPES = Literal["simple_rnn", "lstm"]
DEFAULT_DRIVE_FILE_ID = "1BHzdUi6-iKz7a3tnZunxcp_Td-7I24C7"


@dataclass
class SequenceBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    train_cutoff: int


def ensure_dataset(csv_path: str | Path = "TSLA.csv", drive_file_id: str = DEFAULT_DRIVE_FILE_ID) -> Path:
    csv_path = Path(csv_path)
    if csv_path.exists():
        return csv_path

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as exc:
        raise FileNotFoundError(
            f"{csv_path} was not found and gdown is not installed. Install gdown or place TSLA.csv at {csv_path.resolve()}."
        ) from exc

    url = f"https://drive.google.com/uc?id={drive_file_id}"
    downloaded = gdown.download(url, str(csv_path), quiet=False)
    if not downloaded:
        raise FileNotFoundError(
            f"Could not download the dataset from Google Drive id {drive_file_id}. Please place TSLA.csv at {csv_path.resolve()}."
        )
    return csv_path


def load_tsla_data(csv_path: str | Path = "TSLA.csv") -> pd.DataFrame:
    csv_path = ensure_dataset(csv_path)
    df = pd.read_csv(csv_path)

    if "Date" not in df.columns:
        raise ValueError("The dataset must include a Date column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    df = df[~df.index.duplicated(keep="first")]
    return clean_time_series_frame(df)


def clean_time_series_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
    object_columns = cleaned.columns.difference(numeric_columns)

    if numeric_columns:
        cleaned[numeric_columns] = cleaned[numeric_columns].apply(pd.to_numeric, errors="coerce")
        cleaned[numeric_columns] = cleaned[numeric_columns].interpolate(method="time", limit_direction="both")
        cleaned[numeric_columns] = cleaned[numeric_columns].ffill().bfill()

    for column in object_columns:
        cleaned[column] = cleaned[column].ffill().bfill()

    cleaned = cleaned.dropna(how="all")
    return cleaned


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"missing_count": df.isna().sum(), "missing_pct": df.isna().mean() * 100}).sort_values(
        "missing_count", ascending=False
    )


def prepare_target_series(df: pd.DataFrame, target_column: str = "Close") -> pd.Series:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")
    target = df[target_column].astype(float).copy()
    target = target.interpolate(method="time", limit_direction="both").ffill().bfill()
    return target


def create_sequences(values: np.ndarray, lookback: int = 60, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("values must have shape (n_samples, 1)")

    x_sequences = []
    y_values = []
    target_positions = []
    last_start = len(values) - lookback - horizon + 1
    for start in range(last_start):
        end = start + lookback
        target_index = end + horizon - 1
        x_sequences.append(values[start:end])
        y_values.append(values[target_index, 0])
        target_positions.append(target_index)

    return np.asarray(x_sequences), np.asarray(y_values), np.asarray(target_positions)


def build_sequence_bundle(
    series: pd.Series,
    lookback: int = 60,
    horizon: int = 1,
    test_size: float = 0.2,
) -> SequenceBundle:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    series = series.astype(float).dropna()
    cutoff = int(len(series) * (1 - test_size))
    if cutoff <= lookback + horizon:
        raise ValueError("Not enough data to build train/test sequences with the requested lookback and horizon.")

    scaler = MinMaxScaler()
    train_values = series.iloc[:cutoff].to_numpy().reshape(-1, 1)
    scaler.fit(train_values)

    scaled_values = scaler.transform(series.to_numpy().reshape(-1, 1))
    x_all, y_all, target_positions = create_sequences(scaled_values, lookback=lookback, horizon=horizon)

    train_mask = target_positions < cutoff
    test_mask = ~train_mask

    x_train = x_all[train_mask]
    y_train = y_all[train_mask]
    x_test = x_all[test_mask]
    y_test = y_all[test_mask]

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Unable to create both train and test sets from the provided series.")

    return SequenceBundle(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, scaler=scaler, train_cutoff=cutoff)


def _build_optimizer(learning_rate: float) -> Adam:
    return Adam(learning_rate=learning_rate)


def build_simple_rnn_model(input_shape: Tuple[int, int], units: int = 64, dropout: float = 0.2, learning_rate: float = 1e-3) -> Sequential:
    model = Sequential(
        [
            SimpleRNN(units, activation="tanh", input_shape=input_shape),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=_build_optimizer(learning_rate), loss="mse", metrics=["mae"])
    return model


def build_lstm_model(input_shape: Tuple[int, int], units: int = 64, dropout: float = 0.2, learning_rate: float = 1e-3) -> Sequential:
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout),
            LSTM(max(16, units // 2)),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=_build_optimizer(learning_rate), loss="mse", metrics=["mae"])
    return model


def train_deep_model(
    model_type: MODEL_TYPES,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 25,
    batch_size: int = 32,
    checkpoint_path: str | Path | None = None,
):
    input_shape = (x_train.shape[1], x_train.shape[2])
    if model_type == "simple_rnn":
        model = build_simple_rnn_model(input_shape=input_shape, units=units, dropout=dropout, learning_rate=learning_rate)
    elif model_type == "lstm":
        model = build_lstm_model(input_shape=input_shape, units=units, dropout=dropout, learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    callbacks = [EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)]
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True))

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    return model, history


def inverse_transform_predictions(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(values).ravel()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_lstm_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: Dict[str, Iterable] | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    cv_splits: int = 3,
):
    if param_grid is None:
        param_grid = {
            "model__units": [32, 64],
            "model__dropout": [0.1, 0.2],
            "model__learning_rate": [1e-3, 5e-4],
        }

    input_shape = (x_train.shape[1], x_train.shape[2])

    def model_builder(units: int = 64, dropout: float = 0.2, learning_rate: float = 1e-3):
        return build_lstm_model(
            input_shape=input_shape,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
        )

    regressor = KerasRegressor(
        model=model_builder,
        units=64,
        dropout=0.2,
        learning_rate=1e-3,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    grid = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=cv_splits),
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    grid.fit(x_train, y_train)
    return grid


def compare_models_for_horizon(
    series: pd.Series,
    horizon: int,
    lookback: int = 60,
    test_size: float = 0.2,
    epochs: int = 25,
    batch_size: int = 32,
    units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    bundle = build_sequence_bundle(series, lookback=lookback, horizon=horizon, test_size=test_size)
    results = {}

    for model_type in ("simple_rnn", "lstm"):
        model, history = train_deep_model(
            model_type=model_type,
            x_train=bundle.x_train,
            y_train=bundle.y_train,
            x_val=bundle.x_test,
            y_val=bundle.y_test,
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_path=Path("models") / f"{model_type}_h{horizon}.keras",
        )
        predictions_scaled = model.predict(bundle.x_test, verbose=0).ravel()
        y_true = inverse_transform_predictions(bundle.scaler, bundle.y_test)
        y_pred = inverse_transform_predictions(bundle.scaler, predictions_scaled)
        results[model_type] = {
            "model": model,
            "history": history,
            "y_true": y_true,
            "y_pred": y_pred,
            "metrics": regression_metrics(y_true, y_pred),
        }

    return results, bundle
