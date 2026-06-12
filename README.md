# 🚀 Tesla Stock Price Prediction using Deep Learning

## 🌐 Live Demo

**Deployed Application:**
[🔗 Try the Live App]https://tesla-stock-price-prediction-lrlrtuy2gwyvpsyfqpyuh4.streamlit.app

---

## 📌 Overview

This project focuses on forecasting Tesla (TSLA) stock closing prices using Deep Learning techniques. Two Recurrent Neural Network architectures—SimpleRNN and LSTM—are implemented, trained, and compared across multiple forecasting horizons.

The project includes data preprocessing, exploratory data analysis (EDA), hyperparameter tuning, model evaluation, and an interactive Streamlit application for real-time prediction and visualization.

---

## ✨ Features

* Tesla stock price forecasting
* SimpleRNN and LSTM model comparison
* Multi-horizon prediction (1-day, 5-day, 10-day)
* Hyperparameter tuning using GridSearchCV
* Interactive Streamlit dashboard
* Historical stock visualization
* Model performance evaluation
* Real-time prediction interface

---

## 🌐 Live Demo

🔗 **Application:** YOUR_DEPLOYED_LINK

📂 **GitHub Repository:** https://github.com/ayu-sh87/Tesla-_Stock-_Price-_Prediction

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Keras
* PyTorch Backend
* Streamlit
* Jupyter Notebook
* Matplotlib

...

## 🎯 Objectives

* Predict Tesla stock closing prices using historical market data.
* Compare the performance of SimpleRNN and LSTM models.
* Analyze short-term and multi-step forecasting accuracy.
* Build an interactive web application for stock price prediction.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* Keras
* PyTorch Backend
* Streamlit
* Jupyter Notebook

---

## 📂 Project Structure

```text
Tesla_Stock_Price_Prediction/
│
├── Tesla_Stock_Price_Prediction.ipynb   # Complete implementation notebook
├── app.py                               # Streamlit web application
├── tsla_utils.py                        # Helper functions and preprocessing
├── report.md                            # Detailed project report
├── requirements.txt                     # Required dependencies
├── TSLA.csv                             # Tesla stock dataset
└── README.md                            # Project documentation
```

---

## 📊 Dataset

The project uses historical Tesla (TSLA) stock market data containing:

* Date
* Open Price
* High Price
* Low Price
* Close Price
* Adjusted Close Price
* Trading Volume

The target variable for prediction is the **Closing Price**.

---

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

* Date parsing and chronological sorting
* Missing value handling
* Time-aware interpolation
* Forward and backward filling
* Feature scaling using MinMaxScaler
* Creation of 60-day lookback sequences

---

## 🧠 Deep Learning Models

### 1. SimpleRNN

A basic recurrent neural network used as the baseline model for stock price forecasting.

### 2. LSTM (Long Short-Term Memory)

An advanced recurrent architecture capable of capturing long-term dependencies in sequential financial data.

---

## 🔍 Hyperparameter Optimization

LSTM hyperparameters are optimized using:

* GridSearchCV
* Cross-validation
* Performance-based model selection

Parameters tuned include:

* Number of units
* Batch size
* Epochs
* Learning configuration

---

## 📈 Forecasting Horizons

The models are trained and evaluated for:

* 1-Day Ahead Prediction
* 5-Day Ahead Prediction
* 10-Day Ahead Prediction

Performance metrics are compared across all forecasting horizons.

---

## 📊 Evaluation Metrics

The following metrics are used to evaluate model performance:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

---

## 💻 Streamlit Application

The project includes an interactive Streamlit dashboard that allows users to:

* Upload Tesla stock data
* Visualize historical trends
* Compare model predictions
* Generate future stock price forecasts
* Analyze model performance

Run the application using:

```bash
streamlit run app.py
```

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Tesla-Stock-Price-Prediction.git
cd Tesla-Stock-Price-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📓 Running the Notebook

Open the notebook in Jupyter Notebook or VS Code:

```bash
jupyter notebook Tesla_Stock_Price_Prediction.ipynb
```

Run all cells sequentially to reproduce the complete workflow.

---

## 📌 Key Findings

* LSTM consistently outperforms SimpleRNN for stock price forecasting.
* Short-term forecasts are significantly more accurate than long-term forecasts.
* Proper sequence generation and data preprocessing substantially improve model performance.
* Financial time-series data remains highly volatile and difficult to predict with perfect accuracy.

---

## 🔮 Future Enhancements

* Integration of Transformer-based architectures.
* Incorporation of financial news sentiment analysis.
* Real-time stock data fetching using APIs.
* Multi-stock portfolio forecasting.
* Model deployment using cloud services.

---

## 👨‍💻 Author

**Ayush Singh**

Deep Learning | Machine Learning | Data Science | Software Development

---

## 📜 License

This project is developed for educational and research purposes.
