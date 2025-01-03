import torch
import numpy as np
import yfinance as yf
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import math


class TransformerPredictor(nn.Module):
    """
    Neural network model using a Transformer architecture for time series prediction.

    The model consists of:
    - Input projection layer
    - Positional encoding
    - Transformer encoder blocks
    - Dense layers for final prediction

    Args:
        input_size (int): Dimension of input features (default: 1)
        hidden_size (int): Dimension of hidden layers (default: 128)
        num_layers (int): Number of transformer encoder layers (default: 4)
        output_size (int): Dimension of output prediction (default: 1)
        n_heads (int): Number of attention heads in transformer (default: 8)
        dropout (float): Dropout rate for regularization (default: 0.2)
    """

    def __init__(self, input_size=1, hidden_size=128, num_layers=4, output_size=1, n_heads=8, dropout=0.2):
        super(TransformerPredictor, self).__init__()
        self.hidden_size = hidden_size

        # Project input features to hidden dimension
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Learnable positional encoding for sequence information
        self.pos_encoder = nn.Parameter(torch.zeros(1, 365, hidden_size))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)

        # Stack of transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final prediction layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_size)
        """
        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoder[:, :x.size(1), :]

        # Pass through transformer
        transformer_out = self.transformer(x)

        # Use final sequence output for prediction
        last_out = transformer_out[:, -1, :]

        # Final dense layers with ReLU activation
        x = F.relu(self.fc1(last_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def load_model() -> TransformerPredictor:
    """
    Initialize and load a pretrained transformer model.

    Returns:
        TransformerPredictor: Loaded model (with pretrained weights if available)
    """
    model = TransformerPredictor(input_size=1, hidden_size=128, num_layers=4, output_size=1, n_heads=8)
    try:
        checkpoint = torch.load("transformer_model.pth", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}. Using initialized weights.")
    return model


def prepare_data_for_prediction(df: pd.DataFrame, sequence_length: int) -> torch.Tensor:
    """
    Prepare time series data for model prediction by creating sequences and scaling.

    Args:
        df (pd.DataFrame): Input dataframe with ticker prices as columns
        sequence_length (int): Length of input sequences to generate

    Returns:
        torch.Tensor: Prepared sequences ready for model input
    """
    X = []
    scaler = StandardScaler()

    for ticker in df.columns:
        try:
            print(f"Processing ticker: {ticker}")
            features = df[[ticker]].dropna().values

            # Ensure enough data points for sequence creation
            if len(features) < sequence_length:
                print(f"Warning: Insufficient data for {ticker} ({len(features)} < {sequence_length} points required)")
                continue

            # Scale features and create sequences
            features = scaler.fit_transform(features)
            for i in range(len(features) - sequence_length):
                X.append(features[i:i + sequence_length])

        except KeyError as e:
            print(f"Error processing {ticker}: {e}")
            continue

    print(f"Generated {len(X)} sequences for prediction")
    return torch.tensor(np.array(X), dtype=torch.float32)


def download_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download historical stock data for multiple tickers.

    Args:
        tickers (List[str]): List of stock ticker symbols
        start_date (str): Start date for data download (YYYY-MM-DD)
        end_date (str): End date for data download (YYYY-MM-DD)

    Returns:
        pd.DataFrame: DataFrame with closing prices for all tickers

    Raises:
        ValueError: If no valid data could be downloaded
    """
    all_data = {}

    for ticker in tqdm(tickers, desc="Downloading historical data"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Close']
            if not data.empty:
                all_data[ticker] = data
            else:
                print(f"No data available for {ticker}")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")

    if not all_data:
        raise ValueError("Failed to download data for any tickers")

    df = pd.concat(all_data, axis=1)
    df.columns = all_data.keys()
    return df


def trading_algorithm(
        tickers: List[str],
        model: TransformerPredictor,
        sequence_length: int,
        start_date: str,
        end_date: str
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate trading predictions using the transformer model.

    Args:
        tickers (List[str]): List of stock tickers to analyze
        model (TransformerPredictor): Trained model for predictions
        sequence_length (int): Length of input sequences
        start_date (str): Start date for analysis
        end_date (str): End date for analysis

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: Model predictions and historical price data
    """
    # Download and prepare data
    df = download_data(tickers, start_date, end_date)
    X = prepare_data_for_prediction(df, sequence_length)

    # Generate predictions
    model.eval()
    dataset = TensorDataset(X)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch_X in tqdm(dataloader, desc="Generating predictions"):
            batch_X = batch_X[0].to(torch.float32)
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().cpu().numpy())

    return np.array(predictions), df


def backtest_strategy(
        predictions: np.ndarray,
        df: pd.DataFrame,
        tickers: List[str],
        sequence_length: int,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001
) -> Tuple[pd.DataFrame, Dict]:
    """
    Backtest the trading strategy using model predictions.

    Implementation details:
    - Trades only on Mondays (weekly rebalancing)
    - Selects top 10 stocks based on model predictions
    - Adjusts positions based on prediction confidence
    - Accounts for transaction costs

    Args:
        predictions (np.ndarray): Model predictions for each stock
        df (pd.DataFrame): Historical price data
        tickers (List[str]): List of stock tickers
        sequence_length (int): Length of input sequences
        initial_capital (float): Starting portfolio value
        transaction_cost (float): Transaction cost as a fraction

    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame with portfolio performance and metrics dictionary
    """
    # Filter for tickers with complete data
    valid_tickers = [ticker for ticker in tickers if ticker in df.columns and df[ticker].notna().all()]
    if not valid_tickers:
        raise ValueError("No tickers with complete data available")

    # Validate prediction dimensions
    num_dates = len(df.index[sequence_length:])
    num_tickers = len(valid_tickers)
    expected_size = num_dates * num_tickers

    if predictions.size > expected_size:
        predictions = predictions[:expected_size]
    if predictions.size != expected_size:
        raise ValueError(f"Prediction size mismatch: got {predictions.size}, expected {expected_size}")

    # Reshape predictions and normalize
    prediction_matrix = predictions.reshape((num_dates, num_tickers))
    prediction_df = pd.DataFrame(prediction_matrix, index=df.index[sequence_length:], columns=valid_tickers)
    prediction_df = prediction_df.div(prediction_df.max(axis=1), axis=0).fillna(0)
    prediction_df = prediction_df.clip(lower=0.01)

    # Initialize portfolio
    portfolio = {'cash': initial_capital, 'positions': {ticker: 0 for ticker in valid_tickers}}
    dates = df.index.unique()
    results = pd.DataFrame(index=dates, columns=['Portfolio_Value', 'Cash', 'Returns'])
    total_trades = 0

    # Get Monday dates for weekly trading
    weekly_dates = [date for date in dates[sequence_length:] if date.weekday() == 0]

    # Run backtest simulation
    for date in dates:
        # Calculate daily portfolio value
        daily_value = portfolio['cash']
        current_prices = df.loc[date, list(portfolio['positions'].keys())]
        position_values = np.array(list(portfolio['positions'].values())) * current_prices
        position_values = np.nan_to_num(position_values)
        daily_value += np.sum(position_values)

        results.loc[date, 'Portfolio_Value'] = daily_value
        results.loc[date, 'Cash'] = portfolio['cash']

        # Execute trades on Mondays
        if date in weekly_dates:
            print(f"\nTrading Day: {date}")
            print(f"Starting Cash: ${portfolio['cash']:,.2f}")

            # Get top stocks based on predictions
            daily_predictions = [
                (ticker, prediction_df.loc[date, ticker])
                for ticker in valid_tickers if ticker in prediction_df.columns
            ]
            daily_predictions.sort(key=lambda x: x[1], reverse=True)
            top_stocks = [ticker for ticker, _ in daily_predictions[:10]]

            # Sell stocks not in top selection
            for ticker in list(portfolio['positions'].keys()):
                if ticker not in top_stocks and ticker in df.columns:
                    current_price = df.loc[date, ticker]
                    shares_to_sell = portfolio['positions'][ticker]
                    revenue = shares_to_sell * current_price * (1 - transaction_cost)
                    portfolio['cash'] += revenue
                    portfolio['positions'][ticker] = 0
                    total_trades += 1
                    if shares_to_sell > 0:
                        print(f"Sold {shares_to_sell} {ticker} @ ${current_price:.2f} = ${revenue:,.2f}")

            # Calculate position sizes for top stocks
            total_available_cash = portfolio['cash']
            weights = {ticker: prediction_df.loc[date, ticker] for ticker in top_stocks}
            total_weight = sum(weights.values())

            if total_weight > 1e-6:
                allocations = {
                    ticker: (weight / total_weight) * total_available_cash
                    for ticker, weight in weights.items()
                }

                # Buy new positions
                for ticker in top_stocks:
                    if ticker in df.columns and not pd.isna(df.loc[date, ticker]):
                        current_price = df.loc[date, ticker]
                        allocation = allocations.get(ticker, 0)
                        desired_shares = allocation // (current_price * (1 + transaction_cost))
                        current_shares = portfolio['positions'].get(ticker, 0)
                        shares_to_buy = max(desired_shares - current_shares, 0)
                        cost = shares_to_buy * current_price * (1 + transaction_cost)

                        if shares_to_buy > 0 and cost <= portfolio['cash']:
                            portfolio['cash'] -= cost
                            portfolio['positions'][ticker] += shares_to_buy
                            total_trades += 1
                            print(f"Bought {shares_to_buy} {ticker} @ ${current_price:.2f} = ${cost:,.2f}")

    # Calculate performance metrics
    results['Portfolio_Value'] = results['Portfolio_Value'].fillna(method='ffill').fillna(initial_capital)
    results['Cash'] = results['Cash'].fillna(method='ffill').fillna(initial_capital)
    results['Returns'] = results['Portfolio_Value'].pct_change()
    results['Cumulative_Returns'] = (1 + results['Returns']).cumprod()

    final_value = results['Portfolio_Value'].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100
    sharpe_ratio = (
        results['Returns'].mean() / results['Returns'].std() * np.sqrt(252)
        if results['Returns'].std() != 0 else 0
    )

    metrics = {
        'Final Portfolio Value': final_value,
        'Total Return (%)': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Total Trades': total_trades
    }

    return results, metrics


def plot_backtest_results(results: pd.DataFrame, metrics: Dict):
    """
    Visualize backtest results and display performance metrics.

    Args:
        results (pd.DataFrame): DataFrame with backtest results
        metrics (Dict): Dictionary of performance metrics
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Portfolio_Value'], label='Portfolio Value')
    plt.title('Trading Strategy Backtest Results')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)

    print("\nPerformance Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key}: ${value:,.2f}" if "Value" in key else f"{key}: {value:.2f}")

    plt.show()

# Main program
if __name__ == "__main__":

    # How much data is used for predictions (higher = more accurate)
    sequence_length = 30

    # Get tickers for the Nasdaq100
    tickers = [ticker.replace('.', '-') for ticker in pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]['Symbol']]

    start_date = '2015-01-01'
    end_date = '2025-01-01'

    # Load model
    model = load_model()

    # Get predictions
    model.eval()
    predictions, df = trading_algorithm(tickers, model, sequence_length, start_date, end_date)

    # Run the backtest
    results, metrics = backtest_strategy(predictions, df, tickers, sequence_length)

    # Plot results
    plot_backtest_results(results, metrics)