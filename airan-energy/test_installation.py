#!/usr/bin/env python3
"""
Installation and Module Test Script

Verifies that all dependencies are installed correctly and modules can be imported.
Run this after installing requirements.txt to ensure everything is set up properly.
"""

import sys
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text:^70}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text):
    """Print error message"""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text):
    """Print warning message"""
    print(f"{YELLOW}⚠{RESET} {text}")


def test_core_dependencies():
    """Test core Python dependencies"""
    print_header("Testing Core Dependencies")

    dependencies = [
        ('jax', 'JAX'),
        ('jaxlib', 'JAXLib'),
        ('flax', 'Flax'),
        ('haiku', 'Haiku'),
        ('optax', 'Optax'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('streamlit', 'Streamlit'),
    ]

    failed = []

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_success(f"{display_name:20} installed")
        except ImportError as e:
            print_error(f"{display_name:20} NOT installed")
            failed.append((display_name, str(e)))

    return len(failed) == 0, failed


def test_jax_functionality():
    """Test JAX functionality"""
    print_header("Testing JAX Functionality")

    try:
        import jax
        import jax.numpy as jnp

        # Test basic operations
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        print_success(f"JAX basic operations work (sum={float(y)})")

        # Test JIT compilation
        @jax.jit
        def f(x):
            return x ** 2

        result = f(jnp.array(2.0))
        print_success(f"JAX JIT compilation works (2^2={float(result)})")

        # Check available devices
        devices = jax.devices()
        print_success(f"JAX devices: {[str(d) for d in devices]}")

        return True

    except Exception as e:
        print_error(f"JAX functionality test failed: {e}")
        return False


def test_project_modules():
    """Test project-specific modules"""
    print_header("Testing Project Modules")

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    modules = [
        ('src.data.dataset_generator', 'Dataset Generator'),
        ('src.models.traffic_forecaster', 'Traffic Forecaster'),
        ('src.models.dqn_controller', 'DQN Controller'),
        ('src.models.energy_calculator', 'Energy Calculator'),
    ]

    failed = []

    for module_path, display_name in modules:
        try:
            __import__(module_path)
            print_success(f"{display_name:25} imported successfully")
        except Exception as e:
            print_error(f"{display_name:25} import failed")
            failed.append((display_name, str(e)))

    return len(failed) == 0, failed


def test_model_creation():
    """Test creating model instances"""
    print_header("Testing Model Creation")

    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    try:
        # Test Traffic Forecaster
        from src.models.traffic_forecaster import TrafficForecasterWrapper
        import jax.numpy as jnp

        forecaster = TrafficForecasterWrapper(
            lookback_window=24,
            forecast_horizon=12,
            input_features=4
        )
        print_success("Traffic Forecaster created successfully")

        # Test forward pass
        dummy_input = jnp.ones((1, 24, 4))
        output = forecaster.predict(forecaster.params, dummy_input)
        print_success(f"Forward pass works (output shape: {output.shape})")

        # Test DQN Controller
        from src.models.dqn_controller import DQNController

        controller = DQNController(state_dim=8, num_actions=4)
        print_success("DQN Controller created successfully")

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
        print_success(f"State encoding works (state shape: {state.shape})")

        # Test Energy Calculator
        from src.models.energy_calculator import EnergyCalculator

        calculator = EnergyCalculator()
        power = calculator.calculate_cell_power(500.0, 1000.0, is_sleeping=False)
        print_success(f"Energy calculation works (power: {power:.2f}W)")

        return True

    except Exception as e:
        print_error(f"Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_generation():
    """Test dataset generation"""
    print_header("Testing Data Generation")

    sys.path.insert(0, str(Path(__file__).parent / 'src'))

    try:
        from src.data.dataset_generator import CellTrafficGenerator

        generator = CellTrafficGenerator(random_seed=42)
        print_success("Dataset generator created")

        # Generate small test dataset
        df = generator.generate_dataset(num_cells=2, num_days=2)
        print_success(f"Dataset generated ({len(df)} records)")

        # Check data structure
        required_cols = ['timestamp', 'cell_id', 'traffic_mbps', 'qos_score']
        has_all_cols = all(col in df.columns for col in required_cols)

        if has_all_cols:
            print_success("Dataset has all required columns")
        else:
            print_error("Dataset missing required columns")
            return False

        return True

    except Exception as e:
        print_error(f"Data generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print(f"{BLUE}╔═══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║     AI-RAN Energy Optimization - Installation Test Suite         ║{RESET}")
    print(f"{BLUE}╚═══════════════════════════════════════════════════════════════════╝{RESET}")

    all_passed = True

    # Test 1: Core dependencies
    passed, failed = test_core_dependencies()
    if not passed:
        all_passed = False
        print_warning(f"\nFailed dependencies: {len(failed)}")
        for name, error in failed:
            print(f"  - {name}: {error}")

    # Test 2: JAX functionality
    if not test_jax_functionality():
        all_passed = False

    # Test 3: Project modules
    passed, failed = test_project_modules()
    if not passed:
        all_passed = False
        print_warning(f"\nFailed modules: {len(failed)}")
        for name, error in failed:
            print(f"  - {name}: {error}")

    # Test 4: Model creation
    if not test_model_creation():
        all_passed = False

    # Test 5: Data generation
    if not test_data_generation():
        all_passed = False

    # Final summary
    print_header("Test Summary")

    if all_passed:
        print_success("All tests passed! Installation is successful.")
        print(f"\n{GREEN}You're ready to run:{RESET}")
        print(f"  {BLUE}python quickstart.py{RESET}")
        print(f"  {BLUE}streamlit run src/dashboard/app.py{RESET}")
        return 0
    else:
        print_error("Some tests failed. Please check the errors above.")
        print(f"\n{YELLOW}Try:{RESET}")
        print(f"  {BLUE}pip install -r requirements.txt{RESET}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    print()
    sys.exit(exit_code)
