"""
Synthetic Cell Traffic Dataset Generator for AI-RAN Energy Optimization

Generates realistic 5G cell site traffic patterns with:
- Daily and weekly seasonality
- Special events and anomalies
- Weather correlation
- Multiple cell types (urban, suburban, rural)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse


class CellTrafficGenerator:
    """Generate synthetic cell site traffic data"""

    def __init__(self, random_seed=42):
        np.random.seed(random_seed)

        # Cell type profiles (avg traffic in Mbps)
        self.cell_profiles = {
            'urban': {'peak': 800, 'base': 200, 'variance': 100},
            'suburban': {'peak': 400, 'base': 80, 'variance': 50},
            'rural': {'peak': 150, 'base': 30, 'variance': 25},
        }

        # Time-of-day patterns (0-23 hours)
        self.hourly_pattern = np.array([
            0.3, 0.2, 0.15, 0.1, 0.1, 0.15,  # 00:00 - 05:00 (night)
            0.3, 0.5, 0.7, 0.8, 0.85, 0.9,   # 06:00 - 11:00 (morning)
            0.95, 0.9, 0.85, 0.8, 0.85, 0.95, # 12:00 - 17:00 (afternoon)
            1.0, 0.95, 0.85, 0.7, 0.55, 0.4   # 18:00 - 23:00 (evening)
        ])

        # Day-of-week patterns (Mon=0, Sun=6)
        self.daily_pattern = np.array([
            0.95, 0.95, 0.95, 0.95, 1.0,  # Mon-Fri
            0.85, 0.75                     # Sat-Sun
        ])

    def generate_base_traffic(self, cell_type, hours):
        """Generate base traffic pattern for a cell"""
        profile = self.cell_profiles[cell_type]

        traffic = np.zeros(hours)
        for h in range(hours):
            hour_of_day = h % 24
            day_of_week = (h // 24) % 7

            # Combine hourly and daily patterns
            multiplier = self.hourly_pattern[hour_of_day] * self.daily_pattern[day_of_week]

            # Base + pattern + noise
            traffic[h] = (
                profile['base'] +
                (profile['peak'] - profile['base']) * multiplier +
                np.random.normal(0, profile['variance'] * 0.1)
            )

        return np.maximum(traffic, 0)  # No negative traffic

    def add_special_events(self, traffic, num_events=5):
        """Add special events (concerts, sports, etc.)"""
        hours = len(traffic)

        for _ in range(num_events):
            # Random event time and duration
            start = np.random.randint(0, hours - 24)
            duration = np.random.randint(2, 6)  # 2-6 hours

            # Traffic surge (1.5x to 3x normal)
            multiplier = np.random.uniform(1.5, 3.0)

            for h in range(start, min(start + duration, hours)):
                traffic[h] *= multiplier

        return traffic

    def add_weather_impact(self, traffic, rainy_days_ratio=0.15):
        """Add weather correlation (bad weather = more indoor usage)"""
        hours = len(traffic)
        days = hours // 24

        # Random rainy days
        rainy_days = np.random.choice(days, int(days * rainy_days_ratio), replace=False)

        for day in rainy_days:
            start_hour = day * 24
            for h in range(start_hour, min(start_hour + 24, hours)):
                # 10-30% traffic increase during rain
                traffic[h] *= np.random.uniform(1.1, 1.3)

        return traffic

    def calculate_qos(self, traffic, cell_capacity):
        """Calculate QoS score based on traffic vs capacity"""
        utilization = traffic / cell_capacity

        # QoS degradation model
        qos = np.where(utilization < 0.7, 100,
               np.where(utilization < 0.85, 95,
               np.where(utilization < 0.95, 85, 70)))

        # Add small random variations
        qos = qos + np.random.normal(0, 2, size=len(qos))
        return np.clip(qos, 0, 100)

    def estimate_num_users(self, traffic, avg_user_throughput=5):
        """Estimate number of active users from traffic"""
        num_users = (traffic / avg_user_throughput).astype(int)
        # Add Poisson noise
        num_users = np.random.poisson(num_users + 1)
        return num_users

    def generate_cell_data(self, cell_id, cell_type, num_days=30):
        """Generate complete data for one cell site"""
        hours = num_days * 24

        # Generate base traffic pattern
        traffic = self.generate_base_traffic(cell_type, hours)

        # Add variations
        traffic = self.add_special_events(traffic, num_events=max(1, num_days // 7))
        traffic = self.add_weather_impact(traffic)

        # Cell capacity (peak traffic * 1.2 for headroom)
        capacity = self.cell_profiles[cell_type]['peak'] * 1.2

        # Calculate derived metrics
        qos = self.calculate_qos(traffic, capacity)
        num_users = self.estimate_num_users(traffic)

        # Create timestamps
        start_date = datetime(2025, 1, 1)
        timestamps = [start_date + timedelta(hours=h) for h in range(hours)]

        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cell_id': cell_id,
            'cell_type': cell_type,
            'traffic_mbps': traffic.round(2),
            'num_users': num_users,
            'qos_score': qos.round(1),
            'capacity_mbps': capacity,
            'utilization': (traffic / capacity * 100).round(1)
        })

        return df

    def generate_dataset(self, num_cells=100, num_days=30,
                        urban_ratio=0.3, suburban_ratio=0.5):
        """Generate dataset for multiple cells"""
        print(f"Generating dataset: {num_cells} cells × {num_days} days")

        # Determine cell types
        num_urban = int(num_cells * urban_ratio)
        num_suburban = int(num_cells * suburban_ratio)
        num_rural = num_cells - num_urban - num_suburban

        cell_types = (
            ['urban'] * num_urban +
            ['suburban'] * num_suburban +
            ['rural'] * num_rural
        )

        # Generate data for each cell
        all_data = []
        for i, cell_type in enumerate(cell_types):
            cell_id = f"CELL_{i:04d}"
            print(f"  Generating {cell_id} ({cell_type})...", end='\r')

            cell_df = self.generate_cell_data(cell_id, cell_type, num_days)
            all_data.append(cell_df)

        # Combine all cells
        df = pd.concat(all_data, ignore_index=True)

        print(f"\nDataset generated: {len(df):,} records")
        print(f"  Urban cells: {num_urban}")
        print(f"  Suburban cells: {num_suburban}")
        print(f"  Rural cells: {num_rural}")

        return df


def generate_neighbor_topology(num_cells):
    """Generate cell neighbor relationships"""
    neighbors = {}

    for i in range(num_cells):
        cell_id = f"CELL_{i:04d}"
        # Each cell has 3-6 neighbors
        num_neighbors = np.random.randint(3, 7)
        neighbor_ids = []

        for _ in range(num_neighbors):
            neighbor_idx = np.random.randint(0, num_cells)
            if neighbor_idx != i:
                neighbor_ids.append(f"CELL_{neighbor_idx:04d}")

        neighbors[cell_id] = list(set(neighbor_ids))  # Remove duplicates

    return neighbors


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic cell traffic dataset')
    parser.add_argument('--num-cells', type=int, default=100, help='Number of cell sites')
    parser.add_argument('--num-days', type=int, default=30, help='Number of days to simulate')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Generate dataset
    generator = CellTrafficGenerator(random_seed=args.seed)
    df = generator.generate_dataset(
        num_cells=args.num_cells,
        num_days=args.num_days
    )

    # Save main dataset
    output_file = output_dir / f'cell_traffic_{args.num_cells}cells_{args.num_days}days.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")

    # Generate and save neighbor topology
    neighbors = generate_neighbor_topology(args.num_cells)
    neighbor_file = output_dir / 'cell_neighbors.csv'
    neighbor_df = pd.DataFrame([
        {'cell_id': cell, 'neighbor_id': neighbor}
        for cell, neighbor_list in neighbors.items()
        for neighbor in neighbor_list
    ])
    neighbor_df.to_csv(neighbor_file, index=False)
    print(f"Saved topology to: {neighbor_file}")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(df.groupby('cell_type')['traffic_mbps'].describe())
    print(f"\nAverage QoS: {df['qos_score'].mean():.1f}")
    print(f"Average Utilization: {df['utilization'].mean():.1f}%")


if __name__ == '__main__':
    main()
