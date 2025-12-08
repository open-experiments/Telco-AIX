"""
Energy Savings Calculator for AI-RAN Optimization

Calculates energy consumption, cost savings, and environmental impact
of cell sleep optimization strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PowerModel:
    """Power consumption model for different cell states"""
    # Power consumption in Watts
    active_full_load: float = 1000.0    # 100% traffic
    active_half_load: float = 700.0     # 50% traffic
    active_idle: float = 500.0          # 0% traffic (but active)
    sleep_light: float = 100.0          # Quick wake capability
    sleep_deep: float = 20.0            # Deep sleep
    transition_power: float = 200.0     # Additional power during wake-up
    transition_duration: float = 2.0    # Minutes to wake up


@dataclass
class CostModel:
    """Cost model for energy and operations"""
    electricity_cost_per_kwh: float = 0.12  # $/kWh
    co2_per_kwh: float = 0.5                # kg CO2 per kWh
    qos_penalty_per_point: float = 10.0     # $ per QoS point below threshold


class EnergyCalculator:
    """Calculate energy consumption and savings"""

    def __init__(self, power_model: PowerModel = None, cost_model: CostModel = None):
        self.power_model = power_model or PowerModel()
        self.cost_model = cost_model or CostModel()

    def calculate_cell_power(self, traffic_mbps: float, capacity_mbps: float, is_sleeping: bool = False) -> float:
        """
        Calculate instantaneous power consumption for a cell

        Args:
            traffic_mbps: Current traffic load in Mbps
            capacity_mbps: Cell capacity in Mbps
            is_sleeping: Whether cell is in sleep mode

        Returns:
            Power consumption in Watts
        """
        if is_sleeping:
            return self.power_model.sleep_light

        # Calculate utilization
        utilization = min(traffic_mbps / capacity_mbps, 1.0)

        # Linear interpolation based on load
        if utilization >= 0.5:
            # Interpolate between half-load and full-load
            power = self.power_model.active_half_load + \
                    (self.power_model.active_full_load - self.power_model.active_half_load) * \
                    (utilization - 0.5) / 0.5
        else:
            # Interpolate between idle and half-load
            power = self.power_model.active_idle + \
                    (self.power_model.active_half_load - self.power_model.active_idle) * \
                    utilization / 0.5

        return power

    def calculate_baseline_energy(
        self,
        traffic_data: pd.DataFrame,
        duration_hours: float = 24.0
    ) -> Dict[str, float]:
        """
        Calculate baseline energy consumption (all cells always on)

        Args:
            traffic_data: DataFrame with columns [timestamp, cell_id, traffic_mbps, capacity_mbps]
            duration_hours: Time period to analyze

        Returns:
            Dictionary with energy metrics
        """
        total_energy_wh = 0.0
        num_samples = 0

        for _, row in traffic_data.iterrows():
            power_w = self.calculate_cell_power(
                traffic_mbps=row['traffic_mbps'],
                capacity_mbps=row['capacity_mbps'],
                is_sleeping=False
            )

            # Assume each row is 1 hour (adjust if different)
            energy_wh = power_w * 1.0  # Wh
            total_energy_wh += energy_wh
            num_samples += 1

        total_energy_kwh = total_energy_wh / 1000.0

        # Calculate costs
        electricity_cost = total_energy_kwh * self.cost_model.electricity_cost_per_kwh
        co2_emissions = total_energy_kwh * self.cost_model.co2_per_kwh

        return {
            'total_energy_kwh': total_energy_kwh,
            'avg_power_w': total_energy_wh / duration_hours if duration_hours > 0 else 0,
            'electricity_cost_usd': electricity_cost,
            'co2_emissions_kg': co2_emissions,
            'num_cell_hours': num_samples
        }

    def calculate_optimized_energy(
        self,
        traffic_data: pd.DataFrame,
        sleep_decisions: pd.DataFrame,
        duration_hours: float = 24.0
    ) -> Dict[str, float]:
        """
        Calculate energy consumption with sleep optimization

        Args:
            traffic_data: DataFrame with traffic data
            sleep_decisions: DataFrame with columns [timestamp, cell_id, action, is_sleeping]
            duration_hours: Time period to analyze

        Returns:
            Dictionary with energy metrics
        """
        # Merge traffic data with sleep decisions
        merged = pd.merge(
            traffic_data,
            sleep_decisions,
            on=['timestamp', 'cell_id'],
            how='left'
        )

        # Fill missing sleep decisions with 'not sleeping'
        merged['is_sleeping'] = merged['is_sleeping'].fillna(False)

        total_energy_wh = 0.0
        transition_count = 0
        previous_state = {}

        for _, row in merged.iterrows():
            # Calculate base power
            power_w = self.calculate_cell_power(
                traffic_mbps=row['traffic_mbps'],
                capacity_mbps=row['capacity_mbps'],
                is_sleeping=row['is_sleeping']
            )

            # Check for state transition
            cell_id = row['cell_id']
            current_sleeping = row['is_sleeping']

            if cell_id in previous_state:
                if previous_state[cell_id] != current_sleeping:
                    # State transition occurred
                    transition_count += 1
                    # Add transition energy cost
                    transition_energy = (
                        self.power_model.transition_power *
                        self.power_model.transition_duration / 60.0  # Convert minutes to hours
                    )
                    total_energy_wh += transition_energy

            previous_state[cell_id] = current_sleeping

            # Add hourly energy
            energy_wh = power_w * 1.0  # Assuming 1-hour intervals
            total_energy_wh += energy_wh

        total_energy_kwh = total_energy_wh / 1000.0

        # Calculate costs
        electricity_cost = total_energy_kwh * self.cost_model.electricity_cost_per_kwh
        co2_emissions = total_energy_kwh * self.cost_model.co2_per_kwh

        return {
            'total_energy_kwh': total_energy_kwh,
            'avg_power_w': total_energy_wh / duration_hours if duration_hours > 0 else 0,
            'electricity_cost_usd': electricity_cost,
            'co2_emissions_kg': co2_emissions,
            'num_transitions': transition_count,
            'num_cell_hours': len(merged)
        }

    def calculate_savings(
        self,
        baseline_metrics: Dict[str, float],
        optimized_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate savings comparing optimized vs baseline

        Returns:
            Dictionary with savings metrics and percentages
        """
        energy_saved_kwh = baseline_metrics['total_energy_kwh'] - optimized_metrics['total_energy_kwh']
        energy_saved_pct = (energy_saved_kwh / baseline_metrics['total_energy_kwh'] * 100
                           if baseline_metrics['total_energy_kwh'] > 0 else 0)

        cost_saved_usd = baseline_metrics['electricity_cost_usd'] - optimized_metrics['electricity_cost_usd']
        cost_saved_pct = (cost_saved_usd / baseline_metrics['electricity_cost_usd'] * 100
                         if baseline_metrics['electricity_cost_usd'] > 0 else 0)

        co2_saved_kg = baseline_metrics['co2_emissions_kg'] - optimized_metrics['co2_emissions_kg']
        co2_saved_pct = (co2_saved_kg / baseline_metrics['co2_emissions_kg'] * 100
                        if baseline_metrics['co2_emissions_kg'] > 0 else 0)

        return {
            'energy_saved_kwh': energy_saved_kwh,
            'energy_saved_pct': energy_saved_pct,
            'cost_saved_usd': cost_saved_usd,
            'cost_saved_pct': cost_saved_pct,
            'co2_saved_kg': co2_saved_kg,
            'co2_saved_pct': co2_saved_pct,
            'num_transitions': optimized_metrics['num_transitions']
        }

    def calculate_qos_impact(
        self,
        traffic_data: pd.DataFrame,
        sleep_decisions: pd.DataFrame,
        qos_threshold: float = 90.0
    ) -> Dict[str, float]:
        """
        Calculate QoS impact of sleep optimization

        Returns:
            Dictionary with QoS metrics
        """
        # Merge data
        merged = pd.merge(
            traffic_data,
            sleep_decisions,
            on=['timestamp', 'cell_id'],
            how='left'
        )

        # Calculate QoS metrics
        baseline_qos = traffic_data['qos_score'].mean()
        optimized_qos = merged['qos_score'].mean()

        # Count violations
        baseline_violations = (traffic_data['qos_score'] < qos_threshold).sum()
        optimized_violations = (merged['qos_score'] < qos_threshold).sum()

        # Calculate penalties
        baseline_penalty = sum(max(0, qos_threshold - qos) for qos in traffic_data['qos_score'])
        optimized_penalty = sum(max(0, qos_threshold - qos) for qos in merged['qos_score'])

        penalty_cost = (optimized_penalty - baseline_penalty) * self.cost_model.qos_penalty_per_point

        return {
            'baseline_avg_qos': baseline_qos,
            'optimized_avg_qos': optimized_qos,
            'qos_change': optimized_qos - baseline_qos,
            'baseline_violations': int(baseline_violations),
            'optimized_violations': int(optimized_violations),
            'additional_violations': int(optimized_violations - baseline_violations),
            'qos_penalty_cost_usd': penalty_cost
        }

    def generate_report(
        self,
        traffic_data: pd.DataFrame,
        sleep_decisions: pd.DataFrame = None,
        duration_hours: float = 24.0
    ) -> Dict[str, any]:
        """
        Generate comprehensive energy and savings report

        Returns:
            Dictionary with all metrics and savings
        """
        # Calculate baseline
        baseline = self.calculate_baseline_energy(traffic_data, duration_hours)

        # If no optimization, return baseline only
        if sleep_decisions is None or sleep_decisions.empty:
            return {
                'baseline': baseline,
                'optimized': None,
                'savings': None,
                'qos_impact': None
            }

        # Calculate optimized
        optimized = self.calculate_optimized_energy(traffic_data, sleep_decisions, duration_hours)

        # Calculate savings
        savings = self.calculate_savings(baseline, optimized)

        # Calculate QoS impact
        qos_impact = self.calculate_qos_impact(traffic_data, sleep_decisions)

        return {
            'baseline': baseline,
            'optimized': optimized,
            'savings': savings,
            'qos_impact': qos_impact
        }

    def print_report(self, report: Dict[str, any]):
        """Print formatted report"""
        print("=" * 70)
        print("AI-RAN ENERGY OPTIMIZATION REPORT")
        print("=" * 70)

        print("\n--- BASELINE (Always-On) ---")
        b = report['baseline']
        print(f"Total Energy: {b['total_energy_kwh']:.2f} kWh")
        print(f"Average Power: {b['avg_power_w']:.2f} W")
        print(f"Electricity Cost: ${b['electricity_cost_usd']:.2f}")
        print(f"CO2 Emissions: {b['co2_emissions_kg']:.2f} kg")

        if report['optimized']:
            print("\n--- OPTIMIZED (Sleep Strategy) ---")
            o = report['optimized']
            print(f"Total Energy: {o['total_energy_kwh']:.2f} kWh")
            print(f"Average Power: {o['avg_power_w']:.2f} W")
            print(f"Electricity Cost: ${o['electricity_cost_usd']:.2f}")
            print(f"CO2 Emissions: {o['co2_emissions_kg']:.2f} kg")
            print(f"State Transitions: {o['num_transitions']}")

            print("\n--- SAVINGS ---")
            s = report['savings']
            print(f"Energy Saved: {s['energy_saved_kwh']:.2f} kWh ({s['energy_saved_pct']:.1f}%)")
            print(f"Cost Saved: ${s['cost_saved_usd']:.2f} ({s['cost_saved_pct']:.1f}%)")
            print(f"CO2 Reduced: {s['co2_saved_kg']:.2f} kg ({s['co2_saved_pct']:.1f}%)")

            print("\n--- QoS IMPACT ---")
            q = report['qos_impact']
            print(f"Baseline Avg QoS: {q['baseline_avg_qos']:.2f}")
            print(f"Optimized Avg QoS: {q['optimized_avg_qos']:.2f}")
            print(f"QoS Change: {q['qos_change']:+.2f}")
            print(f"Additional Violations: {q['additional_violations']}")
            if q['qos_penalty_cost_usd'] > 0:
                print(f"QoS Penalty Cost: ${q['qos_penalty_cost_usd']:.2f}")

            print("\n--- NET BENEFIT ---")
            net_savings = s['cost_saved_usd'] - max(0, q['qos_penalty_cost_usd'])
            print(f"Total Cost Savings: ${net_savings:.2f}")

        print("=" * 70)


# Example usage
if __name__ == '__main__':
    # Create sample data
    np.random.seed(42)

    traffic_data = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=24, freq='h'),
        'cell_id': ['CELL_0001'] * 24,
        'traffic_mbps': np.random.uniform(100, 800, 24),
        'capacity_mbps': [1000] * 24,
        'qos_score': np.random.uniform(85, 100, 24)
    })

    sleep_decisions = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=24, freq='h'),
        'cell_id': ['CELL_0001'] * 24,
        'action': np.random.choice([0, 1, 2, 3], 24),
        'is_sleeping': np.random.choice([False, True], 24, p=[0.7, 0.3])
    })

    # Calculate and print report
    calculator = EnergyCalculator()
    report = calculator.generate_report(traffic_data, sleep_decisions, duration_hours=24)
    calculator.print_report(report)
