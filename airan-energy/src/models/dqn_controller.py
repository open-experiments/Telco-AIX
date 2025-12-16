"""
Deep Q-Network (DQN) Controller for Cell Sleep Optimization

Implements a DQN agent that learns optimal cell on/off policies to minimize
energy consumption while maintaining QoS requirements.
"""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from collections import deque
import numpy as np
from typing import Tuple, NamedTuple


class Experience(NamedTuple):
    """Single experience tuple for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """Sample random batch of experiences"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def create_q_network(num_actions: int):
    """Create Q-network using Haiku"""

    def network(state):
        """
        Q-network architecture
        Input: state vector
        Output: Q-values for each action
        """
        mlp = hk.Sequential([
            hk.Linear(128), jax.nn.relu,
            hk.Linear(128), jax.nn.relu,
            hk.Linear(64), jax.nn.relu,
            hk.Linear(num_actions)
        ])
        return mlp(state)

    return hk.without_apply_rng(hk.transform(network))


class DQNController:
    """
    DQN-based cell sleep controller

    State space:
    - Current traffic load (normalized)
    - Predicted traffic (next hour)
    - Current QoS score
    - Number of active neighbor cells
    - Time of day (sin/cos encoded)
    - Day of week (sin/cos encoded)

    Action space:
    0: Keep cell ON
    1: Sleep for 30 minutes
    2: Sleep for 1 hour
    3: Sleep for 2 hours

    Reward:
    - Energy saved (positive)
    - QoS penalty (negative if QoS < threshold)
    - Transition cost (negative for sleep/wake transitions)
    """

    # Action definitions
    ACTIONS = {
        0: {'name': 'ON', 'sleep_duration': 0},
        1: {'name': 'SLEEP_30MIN', 'sleep_duration': 0.5},
        2: {'name': 'SLEEP_1H', 'sleep_duration': 1.0},
        3: {'name': 'SLEEP_2H', 'sleep_duration': 2.0}
    }

    # Energy parameters (in Watts)
    POWER_ACTIVE = 1000
    POWER_SLEEP = 100
    POWER_TRANSITION = 200  # Additional cost during wake-up

    def __init__(
        self,
        state_dim: int = 8,
        num_actions: int = 4,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        seed: int = 42
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # Initialize networks
        self.q_network = create_q_network(num_actions)
        self.target_network = create_q_network(num_actions)

        # Initialize parameters
        self.rng = jax.random.PRNGKey(seed)
        dummy_state = jnp.ones((1, state_dim))

        self.rng, init_rng = jax.random.split(self.rng)
        self.params = self.q_network.init(init_rng, dummy_state)
        self.target_params = self.params

        # Optimizer
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # Training metrics
        self.steps = 0
        self.episode_rewards = []

    def encode_state(
        self,
        traffic: float,
        predicted_traffic: float,
        qos: float,
        num_active_neighbors: int,
        hour_of_day: int,
        day_of_week: int,
        is_sleeping: bool,
        sleep_remaining: float
    ) -> np.ndarray:
        """
        Encode environment state into feature vector

        Returns: state vector of shape (state_dim,)
        """
        # Normalize traffic (assuming max 1000 Mbps)
        traffic_norm = traffic / 1000.0
        predicted_traffic_norm = predicted_traffic / 1000.0

        # Normalize QoS (0-100 -> 0-1)
        qos_norm = qos / 100.0

        # Normalize neighbor count (assuming max 6 neighbors)
        neighbors_norm = num_active_neighbors / 6.0

        # Cyclical encoding for time
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        state = np.array([
            traffic_norm,
            predicted_traffic_norm,
            qos_norm,
            neighbors_norm,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos
        ], dtype=np.float32)

        return state

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(0, self.num_actions)
        else:
            # Greedy action (exploitation)
            state_batch = jnp.expand_dims(state, axis=0)
            q_values = self.q_network.apply(self.params, state_batch)
            return int(jnp.argmax(q_values[0]))

    def calculate_reward(
        self,
        action: int,
        qos: float,
        qos_threshold: float = 90.0,
        previous_action: int = 0
    ) -> float:
        """
        Calculate reward for state-action pair

        Reward components:
        1. Energy saved
        2. QoS penalty
        3. Transition cost
        """
        # Energy saved (compared to always-on baseline)
        sleep_duration = self.ACTIONS[action]['sleep_duration']
        energy_saved = (self.POWER_ACTIVE - self.POWER_SLEEP) * sleep_duration
        energy_reward = energy_saved / 1000.0  # Normalize

        # QoS penalty
        if qos < qos_threshold:
            qos_penalty = -10.0 * (qos_threshold - qos) / qos_threshold
        else:
            qos_penalty = 0.0

        # Transition cost (penalize frequent sleep/wake cycles)
        transition_cost = 0.0
        if action != previous_action:
            if action > 0 or previous_action > 0:  # Any sleep transition
                transition_cost = -2.0

        total_reward = energy_reward + qos_penalty + transition_cost

        return total_reward

    @jax.jit
    def loss_fn(self, params, target_params, states, actions, rewards, next_states, dones, gamma):
        """Compute DQN loss (Temporal Difference error)"""
        # Current Q-values
        q_values = self.q_network.apply(params, states)
        q_values = q_values[jnp.arange(len(actions)), actions]

        # Target Q-values (Double DQN)
        next_q_values = self.q_network.apply(params, next_states)
        next_actions = jnp.argmax(next_q_values, axis=1)

        target_q_values = self.target_network.apply(target_params, next_states)
        target_q_values = target_q_values[jnp.arange(len(next_actions)), next_actions]

        # TD target
        targets = rewards + gamma * target_q_values * (1 - dones)

        # Mean squared error
        loss = jnp.mean((q_values - jax.lax.stop_gradient(targets)) ** 2)

        return loss

    @jax.jit
    def train_step(self, params, target_params, opt_state, states, actions, rewards, next_states, dones):
        """Single training step"""
        loss, grads = jax.value_and_grad(self.loss_fn)(
            params, target_params, states, actions, rewards, next_states, dones, self.gamma
        )

        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    def train(self, batch_size: int = 32):
        """Train on a batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to JAX arrays
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones)

        # Training step
        self.params, self.opt_state, loss = self.train_step(
            self.params, self.target_params, self.opt_state,
            states, actions, rewards, next_states, dones
        )

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_params = self.params

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return float(loss)

    def add_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.add(experience)

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        import pickle
        checkpoint = {
            'params': self.params,
            'target_params': self.target_params,
            'opt_state': self.opt_state,
            'steps': self.steps,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards
        }
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"DQN checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        import pickle
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
            self.params = checkpoint['params']
            self.target_params = checkpoint['target_params']
            self.opt_state = checkpoint['opt_state']
            self.steps = checkpoint['steps']
            self.epsilon = checkpoint['epsilon']
            self.episode_rewards = checkpoint.get('episode_rewards', [])
        print(f"DQN checkpoint loaded from {path}")


# Example usage
if __name__ == '__main__':
    # Create DQN controller
    controller = DQNController(state_dim=8, num_actions=4)

    # Test state encoding
    state = controller.encode_state(
        traffic=500.0,
        predicted_traffic=600.0,
        qos=95.0,
        num_active_neighbors=4,
        hour_of_day=14,
        day_of_week=2,
        is_sleeping=False,
        sleep_remaining=0.0
    )
    print(f"Encoded state shape: {state.shape}")
    print(f"State values: {state}")

    # Test action selection
    action = controller.select_action(state, training=True)
    print(f"\nSelected action: {action} ({controller.ACTIONS[action]['name']})")

    # Test reward calculation
    reward = controller.calculate_reward(action=1, qos=95.0, previous_action=0)
    print(f"Reward: {reward:.2f}")

    print("\nDQN Controller initialized successfully!")
