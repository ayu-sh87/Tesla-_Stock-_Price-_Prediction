from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from tsla_utils import (
    build_sequence_bundle,
    compare_models_for_horizon,
    inverse_transform_predictions,
    load_tsla_data,
    prepare_target_series,
    regression_metrics,
    run_lstm_grid_search,
    train_deep_model,
)


st.set_page_config(page_title="Tesla Stock Price Prediction", layout="wide")
st.title("Tesla Stock Price Prediction")
st.caption("SimpleRNN and LSTM forecasting for 1-day, 5-day, and 10-day closing-price predictions.")


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_tsla_data("TSLA.csv")


def plot_prediction_series(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true, label="Actual", color="#111111", linewidth=1.5)
    ax.plot(y_pred, label="Predicted", color="#0F62FE", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Test observations")
    ax.set_ylabel("Close")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


with st.spinner("Loading Tesla data..."):
    data = load_data()
    target = prepare_target_series(data, target_column="Close")

left, right = st.columns([1.15, 0.85])

with left:
    st.subheader("Dataset preview")
    st.dataframe(data.head(10), width="stretch")

with right:
    st.subheader("Quick facts")
    st.metric("Rows", f"{len(data):,}")
    st.metric("Columns", f"{len(data.columns):,}")
    st.metric("Missing values", f"{int(data.isna().sum().sum()):,}")

with st.expander("Visualize the closing price trend", expanded=True):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(target.index, target.values, color="#0F62FE", linewidth=1.4)
    ax.set_title("Tesla Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.sidebar.header("Model settings")
model_type = st.sidebar.selectbox("Model", ["simple_rnn", "lstm"])
horizon = st.sidebar.selectbox("Forecast horizon (days)", [1, 5, 10])
lookback = st.sidebar.slider("Lookback window", min_value=20, max_value=120, value=60, step=5)
test_size = st.sidebar.slider("Test split", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
epochs = st.sidebar.slider("Epochs", min_value=5, max_value=50, value=20, step=5)
batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)
units = st.sidebar.slider("Recurrent units", min_value=16, max_value=128, value=64, step=16)
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
learning_rate = st.sidebar.selectbox("Learning rate", [0.001, 0.0005, 0.0001], index=0)
use_grid_search = st.sidebar.checkbox("Tune LSTM with GridSearchCV", value=(model_type == "lstm" and horizon == 1))
fast_comparison = st.sidebar.checkbox("Fast comparison mode", value=True)

run_single_model = st.sidebar.button("Train selected model")
run_full_comparison = st.sidebar.button("Run full 1/5/10-day comparison")

if run_single_model:
    with st.spinner("Preparing sequences and training the model..."):
        bundle = build_sequence_bundle(target, lookback=lookback, horizon=horizon, test_size=test_size)
        chosen_units = units
        chosen_dropout = dropout
        chosen_learning_rate = learning_rate

        if model_type == "lstm" and use_grid_search:
            grid = run_lstm_grid_search(bundle.x_train, bundle.y_train, epochs=max(5, epochs // 2), batch_size=batch_size)
            chosen_units = int(grid.best_params_.get("model__units", chosen_units))
            chosen_dropout = float(grid.best_params_.get("model__dropout", chosen_dropout))
            chosen_learning_rate = float(grid.best_params_.get("model__learning_rate", chosen_learning_rate))
            st.write("Best grid-search parameters:")
            st.json(grid.best_params_)

        model, history = train_deep_model(
            model_type=model_type,
            x_train=bundle.x_train,
            y_train=bundle.y_train,
            x_val=bundle.x_test,
            y_val=bundle.y_test,
            units=chosen_units,
            dropout=chosen_dropout,
            learning_rate=chosen_learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            checkpoint_path=f"models/{model_type}_h{horizon}.keras",
        )

        predictions_scaled = model.predict(bundle.x_test, verbose=0).ravel()
        y_true = inverse_transform_predictions(bundle.scaler, bundle.y_test)
        y_pred = inverse_transform_predictions(bundle.scaler, predictions_scaled)
        metrics = regression_metrics(y_true, y_pred)

        metric_cols = st.columns(5)
        metric_cols[0].metric("MSE", f"{metrics['mse']:.4f}")
        metric_cols[1].metric("RMSE", f"{metrics['rmse']:.4f}")
        metric_cols[2].metric("MAE", f"{metrics['mae']:.4f}")
        metric_cols[3].metric("MAPE", f"{metrics['mape']:.4f}")
        metric_cols[4].metric("R2", f"{metrics['r2']:.4f}")

        st.subheader("Actual vs predicted")
        plot_prediction_series(y_true, y_pred, f"{model_type.upper()} | Horizon {horizon} day(s)")

        st.subheader("Training history")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(history.history["loss"], label="Train loss")
        if "val_loss" in history.history:
            ax.plot(history.history["val_loss"], label="Validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title("Model Training Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Prediction sample")
        result_frame = pd.DataFrame({"actual": y_true, "predicted": y_pred})
        st.dataframe(result_frame.head(20), width="stretch")

if run_full_comparison:
    with st.spinner("Running SimpleRNN and LSTM comparison across all horizons..."):
        comparison_lookback = 30 if fast_comparison else lookback
        comparison_epochs = 3 if fast_comparison else max(10, epochs)
        comparison_batch_size = 128 if fast_comparison else batch_size
        rows = []
        comparison_payload = {}
        for horizon_choice in [1, 5, 10]:
            results, _bundle = compare_models_for_horizon(
                target,
                horizon=horizon_choice,
                lookback=comparison_lookback,
                test_size=test_size,
                epochs=comparison_epochs,
                batch_size=comparison_batch_size,
                units=units,
                dropout=dropout,
                learning_rate=learning_rate,
            )
            comparison_payload[horizon_choice] = results
            for model_name, payload in results.items():
                rows.append({"horizon": horizon_choice, "model": model_name, **payload["metrics"]})

        comparison_df = pd.DataFrame(rows).sort_values(["horizon", "mse"]).reset_index(drop=True)
        st.subheader("Model comparison table")
        st.dataframe(comparison_df, width="stretch")

        selected_horizon = st.selectbox("View horizon plots", [1, 5, 10], key="comparison_horizon")
        plot_cols = st.columns(2)
        for idx, model_name in enumerate(["simple_rnn", "lstm"]):
            payload = comparison_payload[selected_horizon][model_name]
            with plot_cols[idx]:
                plot_prediction_series(
                    payload["y_true"],
                    payload["y_pred"],
                    f"{model_name.upper()} | Horizon {selected_horizon} day(s)",
                )
