"""
Streamlit Dashboard for AI-RAN Energy Optimization

Real-time monitoring of energy savings and network performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.energy_calculator import EnergyCalculator


def load_sample_data():
    """Load or generate sample data for dashboard"""
    # Generate 24 hours of data
    np.random.seed(42)
    timestamps = pd.date_range('2025-01-01', periods=24, freq='h')

    df = pd.DataFrame({
        'timestamp': timestamps,
        'cell_id': 'CELL_0001',
        'traffic_mbps': np.random.uniform(100, 800, 24),
        'capacity_mbps': 1000,
        'qos_score': np.random.uniform(85, 100, 24),
        'predicted_traffic': np.random.uniform(100, 800, 24),
    })

    # Simulate sleep decisions
    df['hour'] = df['timestamp'].dt.hour
    df['is_sleeping'] = df['hour'].isin(range(0, 6))
    df['action'] = df['is_sleeping'].apply(lambda x: 2 if x else 0)

    return df


def plot_traffic_and_predictions(df):
    """Plot traffic and predictions over time"""
    fig = go.Figure()

    # Actual traffic
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['traffic_mbps'],
        name='Actual Traffic',
        line=dict(color='blue', width=2)
    ))

    # Predicted traffic
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['predicted_traffic'],
        name='Predicted Traffic',
        line=dict(color='orange', width=2, dash='dash')
    ))

    # Highlight sleep periods
    sleep_periods = df[df['is_sleeping']]
    for _, row in sleep_periods.iterrows():
        fig.add_vrect(
            x0=row['timestamp'],
            x1=row['timestamp'] + timedelta(hours=1),
            fillcolor="gray",
            opacity=0.2,
            line_width=0,
            annotation_text="Sleep",
            annotation_position="top left"
        )

    fig.update_layout(
        title='Cell Traffic Over Time',
        xaxis_title='Time',
        yaxis_title='Traffic (Mbps)',
        hovermode='x unified',
        height=400
    )

    return fig


def plot_energy_comparison(baseline, optimized):
    """Plot energy consumption comparison"""
    categories = ['Energy (kWh)', 'Cost ($)', 'CO2 (kg)']
    baseline_values = [
        baseline['total_energy_kwh'],
        baseline['electricity_cost_usd'],
        baseline['co2_emissions_kg']
    ]
    optimized_values = [
        optimized['total_energy_kwh'],
        optimized['electricity_cost_usd'],
        optimized['co2_emissions_kg']
    ]

    fig = go.Figure(data=[
        go.Bar(name='Baseline (Always On)', x=categories, y=baseline_values, marker_color='red'),
        go.Bar(name='Optimized (Sleep)', x=categories, y=optimized_values, marker_color='green')
    ])

    fig.update_layout(
        title='Energy Consumption Comparison',
        yaxis_title='Value',
        barmode='group',
        height=400
    )

    return fig


def plot_qos_over_time(df):
    """Plot QoS score over time"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['qos_score'],
        name='QoS Score',
        line=dict(color='green', width=2),
        fill='tozeroy'
    ))

    # Add threshold line
    fig.add_hline(
        y=90,
        line_dash="dash",
        line_color="red",
        annotation_text="QoS Threshold (90%)"
    )

    fig.update_layout(
        title='Quality of Service Over Time',
        xaxis_title='Time',
        yaxis_title='QoS Score',
        yaxis_range=[0, 100],
        hovermode='x unified',
        height=400
    )

    return fig


def main():
    """Main dashboard app"""

    st.set_page_config(
        page_title="AI-RAN Energy Optimizer",
        page_icon="⚡",
        layout="wide"
    )

    # Title
    st.title("⚡ AI-RAN Energy Efficiency Dashboard")
    st.markdown("Real-time monitoring of cell sleep optimization powered by JAX")

    # Sidebar
    st.sidebar.header("Configuration")
    num_cells = st.sidebar.slider("Number of Cells", 1, 100, 10)
    qos_threshold = st.sidebar.slider("QoS Threshold", 80, 100, 90)
    optimization_mode = st.sidebar.selectbox(
        "Optimization Mode",
        ["Rule-Based", "DQN", "Hybrid"]
    )

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Load data
    df = load_sample_data()

    # Calculate energy metrics
    calculator = EnergyCalculator()
    report = calculator.generate_report(
        df[['timestamp', 'cell_id', 'traffic_mbps', 'capacity_mbps', 'qos_score']],
        df[['timestamp', 'cell_id', 'action', 'is_sleeping']],
        duration_hours=24
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Energy Saved",
            f"{report['savings']['energy_saved_kwh']:.2f} kWh",
            f"{report['savings']['energy_saved_pct']:.1f}%",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Cost Saved",
            f"${report['savings']['cost_saved_usd']:.2f}",
            f"{report['savings']['cost_saved_pct']:.1f}%",
            delta_color="normal"
        )

    with col3:
        st.metric(
            "CO2 Reduced",
            f"{report['savings']['co2_saved_kg']:.2f} kg",
            f"{report['savings']['co2_saved_pct']:.1f}%",
            delta_color="normal"
        )

    with col4:
        qos_change = report['qos_impact']['qos_change']
        st.metric(
            "Avg QoS",
            f"{report['qos_impact']['optimized_avg_qos']:.1f}",
            f"{qos_change:+.1f}",
            delta_color="normal" if qos_change >= 0 else "inverse"
        )

    # Charts
    st.subheader("📊 Traffic Analysis")
    st.plotly_chart(plot_traffic_and_predictions(df), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚡ Energy Comparison")
        st.plotly_chart(
            plot_energy_comparison(report['baseline'], report['optimized']),
            use_container_width=True
        )

    with col2:
        st.subheader("📶 Quality of Service")
        st.plotly_chart(plot_qos_over_time(df), use_container_width=True)

    # Detailed report
    with st.expander("📄 Detailed Energy Report"):
        st.subheader("Baseline (Always-On)")
        st.json({
            "Energy (kWh)": f"{report['baseline']['total_energy_kwh']:.2f}",
            "Cost ($)": f"{report['baseline']['electricity_cost_usd']:.2f}",
            "CO2 (kg)": f"{report['baseline']['co2_emissions_kg']:.2f}",
            "Avg Power (W)": f"{report['baseline']['avg_power_w']:.2f}"
        })

        st.subheader("Optimized (Sleep Strategy)")
        st.json({
            "Energy (kWh)": f"{report['optimized']['total_energy_kwh']:.2f}",
            "Cost ($)": f"{report['optimized']['electricity_cost_usd']:.2f}",
            "CO2 (kg)": f"{report['optimized']['co2_emissions_kg']:.2f}",
            "Avg Power (W)": f"{report['optimized']['avg_power_w']:.2f}",
            "Transitions": f"{report['optimized']['num_transitions']}"
        })

    # Data table
    with st.expander("📋 Raw Data"):
        st.dataframe(df, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using **JAX**, **Flax**, and **Streamlit** | "
        "[Telco-AIX](https://github.com/tme-osx/Telco-AIX)"
    )


if __name__ == '__main__':
    main()
