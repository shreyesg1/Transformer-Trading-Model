# Transformer-Based Stock Trading Algorithm

This project uses a Transformer-based neural network to predict stock prices and optimize trading strategies. It includes scripts for training a model with historical stock data and executing a trading algorithm that selects top-performing stocks, adjusts portfolio allocation, and evaluates performance through backtesting.

## Overview

This project leverages a Transformer-based neural network for stock price prediction and trading strategy optimization. It consists of two main components:

1. **Training Script**: Prepares data, trains a Transformer model on historical stock data, and evaluates the model's performance.
2. **Trading Script**: Uses the trained model to make predictions, execute a trading algorithm, and backtest the strategy.

## Features

### Training Script

- **Data Preparation**: Downloads historical stock data from Yahoo Finance, calculates technical indicators (RSI, Bollinger Bands, MACD), and normalizes the data.
- **Transformer Architecture**: Implements a custom Transformer model with positional encoding, multiple dense layers, and skip connections for robust predictions.
- **Training**:
  - Uses the `MSELoss` criterion and Adam optimizer with learning rate scheduling.
  - Supports gradient clipping and regularization for stability.
- **Output**: Saves the trained model for use in the trading script.

### Trading Script

- **Model Inference**: Loads the trained Transformer model to generate predictions for stock prices based on recent data sequences.
- **Trading Algorithm**:
  - Predicts future stock prices.
  - Selects top-performing stocks based on predictions.
  - Executes weekly trades to adjust portfolio allocation.
- **Backtesting**:
  - Simulates portfolio performance based on trading decisions.
  - Calculates key metrics: total return, Sharpe ratio, and total trades.
- **Visualization**: Plots portfolio value over time.

## Requirements

- Python 3.8+
- Libraries:
  - `torch`
  - `numpy`
  - `pandas`
  - `yfinance`
  - `scikit-learn`
  - `matplotlib`
  - `tqdm`
  - `ta` (for technical indicators)

## Installation

1. Clone the repository or copy the scripts.
2. Install required libraries:
   ```bash
   pip install torch numpy pandas yfinance scikit-learn matplotlib tqdm ta
   ```

## Training the Model

1. **Prepare Data**:

   - The training script downloads historical stock data and computes technical indicators.
   - Adjust `start_date` and `end_date` to specify the training period.

2. **Train the Model**:

   - Run `training.py` to train the Transformer model.
   - The model is saved as `momentum_transformer1.pth` upon completion.

3. **Key Parameters**:

   - `sequence_length`: Length of the input sequence (default: 60).
   - `learning_rate`: Learning rate for the optimizer (default: 5e-6).
   - `epochs`: Number of training epochs (default: 50).

## Using the Trading Algorithm

1. **Load Model**:

   - The trading script loads the trained Transformer model using `load_model()`.

2. **Download Data**:

   - Specify `start_date` and `end_date` to download historical stock data for predictions.

3. **Run Backtest**:

   - Execute the trading script to backtest the strategy on historical data.
   - The algorithm predicts stock prices, selects top stocks, and simulates portfolio performance.

4. **Visualization**:

   - Results include a plot of portfolio value and metrics such as total return and Sharpe ratio.

## Example Usage

### Training

```bash
python training.py
```

### Trading and Backtesting

```bash
python trading.py
```

## Key Components

### Transformer Model Architecture

The Transformer model includes:

- **Input Embedding**: Projects input features to a higher-dimensional space.
- **Positional Encoding**: Adds temporal context to input sequences.
- **Transformer Encoder**: Captures temporal dependencies using multi-head self-attention.
- **Dense Layers**: Maps encoded features to predictions with non-linear transformations.

### Backtesting Metrics

- **Total Return**: Measures overall portfolio growth.
- **Sharpe Ratio**: Evaluates risk-adjusted performance.
- **Total Trades**: Tracks the number of trades executed during backtesting.

## Notes

- Ensure stock tickers are valid and data is available on Yahoo Finance.
- Handle missing data and scaling issues during data preparation.
- Experiment with hyperparameters to optimize model performance.

## Disclaimer

This project is for educational purposes only. It does not constitute financial advice or guarantee profitable trading outcomes. Use at your own risk.

