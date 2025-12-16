"""
Training script for Traffic Forecasting Model

Trains the JAX-based traffic forecaster on cell site data.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.traffic_forecaster import TrafficForecasterWrapper, create_sequences


class TrafficDataLoader:
    """Data loader for traffic forecasting"""

    def __init__(
        self,
        data_path: str,
        lookback_window: int = 168,
        forecast_horizon: int = 24,
        batch_size: int = 32,
        train_split: float = 0.8
    ):
        self.data_path = data_path
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.train_split = train_split

        # Load and preprocess data
        self.load_data()

    def load_data(self):
        """Load and preprocess cell traffic data"""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)

        # Sort by cell and timestamp
        df = df.sort_values(['cell_id', 'timestamp'])

        # Select features
        feature_cols = ['traffic_mbps', 'num_users', 'qos_score', 'utilization']

        # Normalize features
        self.feature_means = df[feature_cols].mean()
        self.feature_stds = df[feature_cols].std()

        df[feature_cols] = (df[feature_cols] - self.feature_means) / self.feature_stds

        # Create sequences for each cell
        all_X, all_y = [], []

        for cell_id in df['cell_id'].unique():
            cell_data = df[df['cell_id'] == cell_id][feature_cols].values

            if len(cell_data) >= self.lookback_window + self.forecast_horizon:
                X, y = create_sequences(
                    cell_data,
                    self.lookback_window,
                    self.forecast_horizon
                )
                all_X.append(X)
                all_y.append(y)

        # Concatenate all sequences
        self.X = np.concatenate(all_X, axis=0)
        self.y = np.concatenate(all_y, axis=0)

        print(f"Created {len(self.X)} sequences")
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

        # Split train/val
        split_idx = int(len(self.X) * self.train_split)
        self.X_train = self.X[:split_idx]
        self.y_train = self.y[:split_idx]
        self.X_val = self.X[split_idx:]
        self.y_val = self.y[split_idx:]

        print(f"Train: {len(self.X_train)}, Val: {len(self.X_val)}")

    def get_train_batches(self):
        """Generate training batches"""
        num_samples = len(self.X_train)
        indices = np.random.permutation(num_samples)

        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]

            yield self.X_train[batch_indices], self.y_train[batch_indices]

    def get_val_batches(self):
        """Generate validation batches"""
        num_samples = len(self.X_val)

        for start_idx in range(0, num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, num_samples)

            yield self.X_val[start_idx:end_idx], self.y_val[start_idx:end_idx]


def train_forecaster(args):
    """Main training function"""

    # Create data loader
    data_loader = TrafficDataLoader(
        data_path=args.data_path,
        lookback_window=args.lookback_window,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size
    )

    # Create model
    forecaster = TrafficForecasterWrapper(
        lookback_window=args.lookback_window,
        forecast_horizon=args.forecast_horizon,
        input_features=4,  # traffic, users, qos, utilization
        learning_rate=args.learning_rate
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)

    for epoch in range(args.epochs):
        # Training
        epoch_train_loss = []

        for batch_X, batch_y in tqdm(
            data_loader.get_train_batches(),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False
        ):
            forecaster.params, forecaster.opt_state, loss = forecaster.train_step(
                forecaster.params,
                forecaster.opt_state,
                batch_X,
                batch_y
            )
            epoch_train_loss.append(float(loss))

        avg_train_loss = np.mean(epoch_train_loss)
        train_losses.append(avg_train_loss)

        # Validation
        epoch_val_loss = []
        for batch_X, batch_y in data_loader.get_val_batches():
            loss = forecaster.loss_fn(forecaster.params, batch_X, batch_y, training=False)
            epoch_val_loss.append(float(loss))

        avg_val_loss = np.mean(epoch_val_loss)
        val_losses.append(avg_val_loss)

        # Print progress
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            forecaster.save_checkpoint(args.output_path)
            print(f"  -> New best model saved (val_loss: {best_val_loss:.6f})")

        # Early stopping
        if epoch > args.patience and val_losses[-args.patience] < min(val_losses[-args.patience:]):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Plot training curves
    if args.plot:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Traffic Forecaster Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(args.output_path.replace('.pkl', '_training_curve.png'))
        print(f"Training curve saved to {args.output_path.replace('.pkl', '_training_curve.png')}")

    # Test predictions
    print("\nTesting predictions...")
    test_X = data_loader.X_val[:5]
    test_y = data_loader.y_val[:5]

    predictions = forecaster.forecast(test_X)

    # Denormalize for display
    test_y_denorm = test_y * data_loader.feature_stds['traffic_mbps'] + data_loader.feature_means['traffic_mbps']
    pred_denorm = predictions * data_loader.feature_stds['traffic_mbps'] + data_loader.feature_means['traffic_mbps']

    print("\nSample predictions (first 5 sequences):")
    for i in range(min(5, len(test_y))):
        mae = np.mean(np.abs(pred_denorm[i] - test_y_denorm[i].squeeze()))
        print(f"  Sample {i+1} - MAE: {mae:.2f} Mbps")

    return forecaster, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train traffic forecasting model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to cell traffic CSV')
    parser.add_argument('--output-path', type=str, default='models/forecaster.pkl',
                       help='Path to save trained model')
    parser.add_argument('--lookback-window', type=int, default=168, help='Lookback window (hours)')
    parser.add_argument('--forecast-horizon', type=int, default=24, help='Forecast horizon (hours)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--plot', action='store_true', help='Plot training curves')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)

    # Train model
    train_forecaster(args)


if __name__ == '__main__':
    main()
