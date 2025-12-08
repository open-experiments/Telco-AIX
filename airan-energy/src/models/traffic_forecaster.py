"""
JAX-based Traffic Forecasting Model for Cell Sites

Implements a Temporal Convolutional Network (TCN) with attention mechanism
for predicting cell site traffic patterns.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence
import optax


class CausalConv1D(nn.Module):
    """Causal 1D convolution for time series (no future information leakage)"""
    features: int
    kernel_size: int
    dilation: int = 1

    @nn.compact
    def __call__(self, x):
        # Calculate padding for causal convolution
        padding = (self.kernel_size - 1) * self.dilation

        # Pad on the left only (causal)
        x = jnp.pad(x, ((0, 0), (padding, 0), (0, 0)), mode='constant')

        # Standard convolution
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            kernel_dilation=(self.dilation,),
            padding='VALID'
        )(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block with causal convolutions and gated activation"""
    features: int
    kernel_size: int
    dilation: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        residual = x

        # First causal conv
        x = CausalConv1D(
            features=self.features,
            kernel_size=self.kernel_size,
            dilation=self.dilation
        )(x)
        x = nn.LayerNorm()(x)

        # Gated activation (like WaveNet)
        tanh_out = jnp.tanh(x)
        sigmoid_out = nn.sigmoid(x)
        x = tanh_out * sigmoid_out

        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Second causal conv
        x = CausalConv1D(
            features=self.features,
            kernel_size=self.kernel_size,
            dilation=self.dilation
        )(x)
        x = nn.LayerNorm()(x)

        # Residual connection (project if needed)
        if residual.shape[-1] != self.features:
            residual = nn.Dense(features=self.features)(residual)

        return x + residual


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, features = x.shape
        embed_dim = self.num_heads * self.head_dim

        # Linear projections
        q = nn.Dense(embed_dim)(x)
        k = nn.Dense(embed_dim)(x)
        v = nn.Dense(embed_dim)(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Scaled dot-product attention
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)

        # Causal mask (prevent attending to future)
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        scores = jnp.where(mask, scores, -1e9)

        attention_weights = nn.softmax(scores, axis=-1)

        # Apply attention to values
        attended = jnp.einsum('bhqk,bkhd->bqhd', attention_weights, v)
        attended = attended.reshape(batch_size, seq_len, embed_dim)

        # Output projection
        output = nn.Dense(features)(attended)
        return output


class TrafficForecaster(nn.Module):
    """
    Traffic Forecasting Model with TCN + Attention

    Architecture:
    - Input: (batch, lookback_window, features)
    - TCN: Multiple residual blocks with increasing dilation
    - Attention: Multi-head attention layer
    - Output: (batch, forecast_horizon, 1)
    """

    tcn_features: int = 64
    num_tcn_blocks: int = 4
    kernel_size: int = 3
    num_attention_heads: int = 8
    attention_head_dim: int = 16
    forecast_horizon: int = 24
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        # Input projection
        x = nn.Dense(features=self.tcn_features)(x)

        # TCN blocks with exponential dilation
        for i in range(self.num_tcn_blocks):
            dilation = 2 ** i
            x = ResidualBlock(
                features=self.tcn_features,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout_rate=self.dropout_rate
            )(x, training=training)

        # Multi-head attention
        x_att = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            head_dim=self.attention_head_dim
        )(x)

        # Combine TCN and attention outputs
        x = x + x_att
        x = nn.LayerNorm()(x)

        # Take last timestep representation
        x = x[:, -1, :]  # (batch, features)

        # Forecast head
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        # Output: forecast for next N hours
        x = nn.Dense(features=self.forecast_horizon)(x)

        # Reshape to (batch, forecast_horizon, 1)
        x = jnp.expand_dims(x, axis=-1)

        return x


class TrafficForecasterWrapper:
    """Wrapper class for training and inference"""

    def __init__(
        self,
        lookback_window: int = 168,  # 7 days
        forecast_horizon: int = 24,   # 24 hours
        input_features: int = 4,      # traffic, users, qos, utilization
        learning_rate: float = 1e-3,
        seed: int = 42
    ):
        self.lookback_window = lookback_window
        self.forecast_horizon = forecast_horizon
        self.input_features = input_features
        self.learning_rate = learning_rate

        # Initialize model
        self.model = TrafficForecaster(forecast_horizon=forecast_horizon)

        # Initialize parameters
        self.rng = jax.random.PRNGKey(seed)
        dummy_input = jnp.ones((1, lookback_window, input_features))
        self.params = self.model.init(self.rng, dummy_input, training=False)

        # Optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

    @jax.jit
    def predict(self, params, x):
        """Make predictions (JIT compiled)"""
        return self.model.apply(params, x, training=False)

    @jax.jit
    def loss_fn(self, params, x, y, training=True):
        """Compute MSE loss"""
        predictions = self.model.apply(params, x, training=training)
        loss = jnp.mean((predictions - y) ** 2)
        return loss

    @jax.jit
    def train_step(self, params, opt_state, x, y):
        """Single training step"""
        loss, grads = jax.value_and_grad(self.loss_fn)(params, x, y, training=True)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in train_loader:
            # Convert to JAX arrays
            x_batch = jnp.array(x_batch)
            y_batch = jnp.array(y_batch)

            # Training step
            self.params, self.opt_state, loss = self.train_step(
                self.params, self.opt_state, x_batch, y_batch
            )

            total_loss += loss
            num_batches += 1

        return total_loss / num_batches

    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in val_loader:
            x_batch = jnp.array(x_batch)
            y_batch = jnp.array(y_batch)

            loss = self.loss_fn(self.params, x_batch, y_batch, training=False)
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches

    def forecast(self, x):
        """Forecast future traffic"""
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x)

        # Add batch dimension if needed
        if x.ndim == 2:
            x = jnp.expand_dims(x, axis=0)

        predictions = self.predict(self.params, x)
        return predictions.squeeze()

    def save_checkpoint(self, path):
        """Save model parameters"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'opt_state': self.opt_state,
                'config': {
                    'lookback_window': self.lookback_window,
                    'forecast_horizon': self.forecast_horizon,
                    'input_features': self.input_features,
                    'learning_rate': self.learning_rate
                }
            }, f)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        """Load model parameters"""
        import pickle
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
            self.params = checkpoint['params']
            self.opt_state = checkpoint['opt_state']
        print(f"Checkpoint loaded from {path}")


def create_sequences(data, lookback, forecast_horizon):
    """
    Create training sequences from time series data

    Args:
        data: (num_timesteps, num_features) array
        lookback: number of past timesteps to use
        forecast_horizon: number of future timesteps to predict

    Returns:
        X: (num_samples, lookback, num_features)
        y: (num_samples, forecast_horizon, 1)
    """
    X, y = [], []

    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + forecast_horizon, 0:1])  # First feature is traffic

    return jnp.array(X), jnp.array(y)


# Example usage
if __name__ == '__main__':
    # Create model
    forecaster = TrafficForecasterWrapper(
        lookback_window=168,  # 7 days
        forecast_horizon=24,   # 24 hours
        input_features=4
    )

    # Test forward pass
    dummy_input = jnp.ones((8, 168, 4))  # Batch of 8
    output = forecaster.predict(forecaster.params, dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model created successfully!")
