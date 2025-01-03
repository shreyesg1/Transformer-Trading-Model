import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
import math
import torch.nn.functional as F


def download_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download and prepare stock data with technical indicators.

    Downloads historical price data and calculates various technical indicators
    including RSI, Bollinger Bands, and MACD for each ticker.

    Args:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: Multi-index DataFrame with stock data and indicators
    """
    all_data = {}
    with tqdm(total=len(tickers), desc="Downloading market data", unit="ticker") as pbar:
        for ticker in tickers:
            try:
                # Download historical data
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)[['Close', 'Volume']]

                # Validate data quality
                if data.empty:
                    print(f"No data available for {ticker}")
                    pbar.update(1)
                    continue

                if data[['Close', 'Volume']].isna().all().all():
                    print(f"Invalid data for {ticker}: All values are NaN")
                    pbar.update(1)
                    continue

                # Calculate price-based indicators
                data = data.dropna()
                data['Returns'] = data['Close'].pct_change()
                data['Log Returns'] = np.log1p(data['Returns'])

                # Technical indicators
                data['RSI'] = RSIIndicator(data['Close'].squeeze()).rsi()

                # Volatility indicators
                bb = BollingerBands(data['Close'].squeeze())
                data['Bollinger High'] = bb.bollinger_hband()
                data['Bollinger Low'] = bb.bollinger_lband()

                # Trend indicators
                macd = MACD(data['Close'].squeeze())
                data['MACD'] = macd.macd()
                data['MACD Signal'] = macd.macd_signal()

                # Clean up and store valid data
                data = data.dropna()
                if not data.empty:
                    all_data[ticker] = data
                else:
                    print(f"No valid data after preprocessing for {ticker}")

            except Exception as e:
                print(f"Failed to process {ticker}: {e}")
            pbar.update(1)

    # Combine all valid data
    if all_data:
        try:
            df = pd.concat(all_data.values(), keys=all_data.keys(), axis=1)
            return df
        except ValueError as e:
            print(f"Failed to combine data: {e}")
            raise
    else:
        print("No valid data collected")
        return pd.DataFrame()


def prepare_data(df: pd.DataFrame, sequence_length: int) -> tuple:
    """
    Prepare and normalize data for model training.

    Creates sequences of historical data and corresponding future prices
    for training the prediction model.

    Args:
        df (pd.DataFrame): Input DataFrame with stock data and indicators
        sequence_length (int): Number of time steps in each input sequence

    Returns:
        tuple: (X, y) where X contains input sequences and y contains target values

    Raises:
        ValueError: If input DataFrame is empty
    """
    if df.empty:
        raise ValueError("Empty DataFrame provided")

    X, y = [], []
    scaler = StandardScaler()

    feature_columns = [
        'Close', 'Volume', 'Returns', 'Log Returns',
        'RSI', 'Bollinger High', 'Bollinger Low',
        'MACD', 'MACD Signal'
    ]

    for ticker in df.columns.levels[0]:
        # Extract and normalize features
        features = df[ticker][feature_columns].dropna().values
        features = scaler.fit_transform(features)

        # Create sequences
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(features[i + sequence_length, 0])  # Predict next Close price

    return (
        torch.tensor(np.array(X), dtype=torch.float32),
        torch.tensor(np.array(y).reshape(-1), dtype=torch.float32)
    )


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for transformer architecture.

    Adds positional information to input embeddings to help the model
    understand sequence order.

    Args:
        d_model (int): Dimension of the model
        dropout (float): Dropout rate
        max_len (int): Maximum sequence length
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MomentumTransformer(nn.Module):
    """
    Transformer-based model for stock price prediction.

    Features:
    - Input embedding layer
    - Positional encoding
    - Multiple transformer encoder layers
    - Skip connections in the feed-forward layers
    - Layer normalization

    Args:
        input_size (int): Number of input features
        d_model (int): Dimension of the model
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        output_size (int): Dimension of output
        dropout (float): Dropout rate
    """

    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_layers: int, output_size: int, dropout: float = 0.1):
        super(MomentumTransformer, self).__init__()

        # Input processing
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Normalization and regularization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Prediction layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.fc3 = nn.Linear(d_model // 4, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_size)
        """
        # Input processing
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.layer_norm1(x)

        # Transformer encoding
        x = self.transformer(x)
        x = self.layer_norm2(x[:, -1, :])  # Use last sequence output

        # Prediction with skip connections
        identity = x
        x = self.dropout(F.relu(self.fc1(x)))
        if identity.size(1) >= x.size(1):
            x = x + identity[:, :x.size(1)]

        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


# Model training configuration
sequence_length = 60  # Number of time steps to look back
learning_rate = 5e-6  # Conservative learning rate for stability
weight_decay = 1e-4  # L2 regularization to prevent overfitting

# Model architecture parameters
d_model = 128  # Dimension of the model
nhead = 8  # Number of attention heads
num_layers = 4  # Number of transformer layers
batch_size = 64  # Training batch size
epochs = 50  # Number of training epochs

# Get list of NASDAQ-100 tickers
ticker_df = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")[4]
tickers = [ticker.replace('.', '-') for ticker in ticker_df['Symbol'].to_list()]

# Training period
start_date = '2015-01-01'
end_date = '2024-12-25'

# Download and prepare data
df = download_data(tickers, start_date, end_date)

if not df.empty:
    # Prepare datasets
    X, y = prepare_data(df, sequence_length)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MomentumTransformer(
        input_size=X.shape[2],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        output_size=1
    )

    # Training setup
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=7,
        factor=0.6,
        verbose=True
    )

    # Training loop
    print("\nStarting model training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}",
                  unit="batch", leave=False, ncols=100) as pbar:
            for batch_X, batch_y in pbar:
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
                optimizer.step()

                # Update progress
                epoch_loss += loss.item()
                pbar.set_postfix(
                    loss=epoch_loss / len(dataloader),
                    eta=f"{pbar.format_dict['elapsed'] / (pbar.n + 1) * (pbar.total - pbar.n):.2f}s"
                )

        # Adjust learning rate
        scheduler.step(epoch_loss / len(dataloader))
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Save trained model
    torch.save(model.state_dict(), "../PythonProject1/momentum_transformer1.pth")
    print("\nModel successfully trained and saved as 'momentum_transformer1.pth'")
else:
    print("Training aborted: No valid data available")