#!/usr/bin/env python3
"""
AI-Driven Forecasting - Model Training Script
Module 5: AIOps Foundations

This script trains Prophet time-series forecasting models on historical
metrics from Prometheus. It learns trends and seasonal patterns to enable
proactive capacity planning.
"""

import os
import sys
from datetime import datetime, timedelta
import warnings
import logging

# Suppress warnings and logging
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

try:
    import pandas as pd
    import numpy as np
    from prometheus_api_client import PrometheusConnect
    import pickle

    # Import Prophet and let it initialize cmdstan
    from prophet import Prophet

except ImportError as e:
    print(f"❌ Error: Missing required package: {e}")
    print("\n💡 Please install packages:")
    print("   pip install -r requirements.txt")
    sys.exit(1)


# Configuration
PROMETHEUS_URL = "http://localhost:9090"
MODEL_DIR = "/root/monitoring/forecasting_models"
TRAINING_HOURS = 1  # Use last 1 hour of data for training (or all available data)
MIN_DATA_POINTS = 20  # Minimum data points required for training (20 * 30s = 10 minutes)

# Metrics to forecast
METRICS_CONFIG = {
    'cpu_usage': {
        'query': '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
        'name': 'CPU Usage (%)',
        'threshold': 80,  # Alert if forecast exceeds this
        'unit': '%'
    },
    'memory_available': {
        'query': 'node_memory_MemAvailable_bytes / 1024 / 1024 / 1024',
        'name': 'Memory Available (GB)',
        'threshold': 3,  # Alert if forecast drops below this
        'unit': 'GB',
        'inverse': True  # Lower values are worse
    },
    'disk_usage': {
        'query': '(1 - node_filesystem_avail_bytes{fstype=~"ext4|xfs|btrfs"} / node_filesystem_size_bytes{fstype=~"ext4|xfs|btrfs"}) * 100',
        'name': 'Disk Usage (%)',
        'threshold': 90,  # Alert if forecast exceeds this
        'unit': '%'
    }
}


def print_header():
    """Print script header."""
    print("\n" + "="*70)
    print("🤖 AI-Driven Forecasting - Model Training")
    print("Module 5: AIOps Foundations")
    print("="*70 + "\n")


def connect_to_prometheus():
    """Connect to Prometheus API."""
    try:
        print(f"🔌 Connecting to Prometheus at {PROMETHEUS_URL}...")
        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

        # Test connection
        prom.check_prometheus_connection()
        print("✅ Connected to Prometheus successfully\n")
        return prom
    except Exception as e:
        print(f"❌ Error connecting to Prometheus: {e}")
        print("💡 Make sure Prometheus is running: docker compose ps")
        sys.exit(1)


def fetch_metric_data(prom, metric_name, query, hours=1):
    """Fetch historical metric data from Prometheus."""
    try:
        print(f"📊 Fetching {metric_name} data...")

        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Query Prometheus
        # Use 30-second step to reduce data points and memory usage
        # (10s = 360 points/hour = high memory, 30s = 120 points/hour = lower memory)
        result = prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step='30s'  # 30-second resolution for lab environment
        )

        if not result or len(result) == 0:
            print(f"⚠️  No data returned for {metric_name}")
            return None

        # Extract values
        values = result[0]['values']

        if len(values) < MIN_DATA_POINTS:
            print(f"⚠️  Insufficient data for {metric_name}: {len(values)} points")
            print(f"    (Need at least {MIN_DATA_POINTS} points)")
            print(f"    💡 Tip: Wait longer or check Prometheus is collecting metrics")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(values, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Remove NaN values
        df = df.dropna()

        if len(df) < MIN_DATA_POINTS:
            print(f"⚠️  Insufficient valid data after cleaning: {len(df)} points")
            return None

        print(f"✅ Fetched {len(df)} data points for {metric_name}")
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Current value: {df['value'].iloc[-1]:.2f}")
        print(f"   Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")

        # Warning for lab environment with limited data
        if len(df) < 500:
            data_hours = len(df) * 30 / 3600  # 30-second intervals
            print(f"\n⚠️  Lab Environment Note: Only {len(df)} data points available (~{data_hours:.1f} hours of data).")
            print("   Prophet requires 2+ weeks for seasonality learning:")
            print("     • Daily patterns: Need 2+ days of data")
            print("     • Weekly patterns: Need 2+ weeks of data")
            print("     • Strong trend detection: Need 1+ week of stable data")
            print(f"\n   Using simplified model for lab demonstration:")
            print("     • growth='logistic' for %  metrics (bounded 0-100%, prevents impossible values)")
            print("     • growth='linear' for other metrics (minimal trend, very restrictive)")
            print("     • seasonality disabled (insufficient data)")
            print("     • 30-second intervals (reduced memory usage)")
            print("     • 3 changepoints max (detect slight trends without wild extrapolation)")
            print(f"\n   For production: Collect 2+ weeks of data @ 10s intervals, then:")
            print("     • daily_seasonality=True, weekly_seasonality=True")
            print("     • changepoint_prior_scale=0.05 (more flexible trend detection)")
            print("     • n_changepoints=25 (default)")
            print("   Continuing with available data...\n")

        return df

    except Exception as e:
        print(f"❌ Error fetching {metric_name}: {e}")
        return None


def prepare_prophet_data(df, metric_config, metric_key):
    """Convert DataFrame to Prophet format (ds, y) with optional cap/floor."""
    prophet_df = pd.DataFrame({
        'ds': df['timestamp'],
        'y': df['value']
    })

    # For percentage metrics, set cap and floor to prevent impossible values
    if metric_config.get('unit') == '%':
        prophet_df['cap'] = 100.0  # CPU/Disk can't exceed 100%

        # Set realistic floor based on metric type
        if metric_key == 'cpu_usage':
            prophet_df['floor'] = 1.0   # CPU can't be 0% on running system (idle processes)
        else:
            prophet_df['floor'] = 0.0   # Disk can theoretically be 0%

    return prophet_df


def train_prophet_model(df, metric_name, metric_config, metric_key):
    """Train Prophet model on historical data."""
    try:
        print(f"\n🧠 Training forecasting model for {metric_name}...")

        # Prepare data for Prophet
        prophet_df = prepare_prophet_data(df, metric_config, metric_key)

        # Determine growth type based on metric
        # For percentage metrics (CPU, Disk), use logistic growth with bounds
        # For unbounded metrics (Memory GB), use linear growth
        is_percentage = metric_config.get('unit') == '%'
        growth_type = 'logistic' if is_percentage else 'linear'

        # Initialize Prophet model
        # For lab environment with limited data (1 hour), use conservative settings
        # Allow minimal trend detection for educational purposes (see slight changes)
        model = Prophet(
            growth=growth_type,          # Logistic for %, linear for others
            daily_seasonality=False,     # Disable (need 2+ days of data)
            weekly_seasonality=False,    # Disable (need 2+ weeks of data)
            yearly_seasonality=False,    # Disable
            n_changepoints=3,            # Minimal changepoints (detect slight trends)
            changepoint_prior_scale=0.001,  # Very restrictive (prevent wild extrapolation)
            interval_width=0.95,         # 95% confidence interval
            mcmc_samples=0,              # Disable MCMC (use MAP estimation, much lower memory)
            uncertainty_samples=100      # Reduce uncertainty samples (default 1000)
        )

        # Train the model
        print(f"   Training on {len(prophet_df)} data points...")

        try:
            model.fit(prophet_df)
        except AttributeError as e:
            # Handle Prophet 1.1.5 stan_backend AttributeError
            if 'stan_backend' in str(e):
                print(f"   Warning: Prophet stan_backend deprecation issue (continuing anyway)")
                # The model may still be trained despite the error
                pass
            else:
                print(f"❌ Error during model.fit(): {e}")
                import traceback
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"❌ Error during model.fit(): {e}")
            import traceback
            traceback.print_exc()
            return None

        # Verify model was trained successfully
        if not hasattr(model, 'params') or model.params is None:
            print(f"❌ Model training failed for {metric_name}: model.params is None")
            print(f"   This usually means cmdstan is not installed or not working properly")
            print(f"   Try: pip install cmdstan")
            return None

        print(f"✅ Model trained successfully for {metric_name}")

        # Analyze learned components
        data_mean = prophet_df['y'].mean()
        data_std = prophet_df['y'].std()
        data_min = prophet_df['y'].min()
        data_max = prophet_df['y'].max()

        # Calculate simple trend
        data_start = prophet_df['y'].iloc[0]
        data_end = prophet_df['y'].iloc[-1]
        trend = data_end - data_start

        print(f"   Learned baseline: mean={data_mean:.2f}, std={data_std:.2f}")
        print(f"   Value range: [{data_min:.2f}, {data_max:.2f}]")
        print(f"   Observed trend: {trend:+.2f} (from {data_start:.2f} to {data_end:.2f})")
        if is_percentage:
            print(f"   Model type: Logistic growth (bounded 0-100%, prevents impossible values)")
        else:
            print(f"   Model type: Linear growth (minimal, restricted by changepoint_prior=0.001)")

        return model

    except Exception as e:
        print(f"❌ Error training model for {metric_name}: {e}")
        return None


def save_model(model, metric_key):
    """Save trained model to disk."""
    try:
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Save model
        model_path = os.path.join(MODEL_DIR, f"{metric_key}_forecast_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"💾 Model saved: {model_path}")
        return model_path

    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return None


def main():
    """Main training workflow."""
    print_header()

    # Connect to Prometheus
    prom = connect_to_prometheus()

    # Track trained models
    trained_models = {}
    failed_metrics = []

    # Train model for each metric
    for metric_key, config in METRICS_CONFIG.items():
        print("\n" + "-"*70)

        # Fetch data
        df = fetch_metric_data(
            prom,
            config['name'],
            config['query'],
            hours=TRAINING_HOURS  # Use last 1 hour (or all available data)
        )

        if df is None:
            failed_metrics.append(metric_key)
            continue

        # Train model
        model = train_prophet_model(df, config['name'], config, metric_key)

        if model is None:
            failed_metrics.append(metric_key)
            continue

        # Save model
        model_path = save_model(model, metric_key)

        if model_path:
            trained_models[metric_key] = {
                'model_path': model_path,
                'name': config['name'],
                'data_points': len(df)
            }

    # Print summary
    print("\n" + "="*70)
    print("📊 Training Summary")
    print("="*70)

    if trained_models:
        print(f"\n✅ Successfully trained {len(trained_models)} model(s):\n")
        for metric_key, info in trained_models.items():
            print(f"   • {info['name']}")
            print(f"     Model: {info['model_path']}")
            print(f"     Training data: {info['data_points']} points")

    if failed_metrics:
        print(f"\n⚠️  Failed to train {len(failed_metrics)} model(s):")
        for metric_key in failed_metrics:
            print(f"   • {METRICS_CONFIG[metric_key]['name']}")
        print(f"\n💡 Common issues:")
        print(f"   - Insufficient historical data (need {MIN_DATA_POINTS}+ points)")
        print(f"   - Prometheus not collecting metrics")
        print(f"   - Data quality issues (all NaN values)")
        print(f"\n   Try waiting longer or check: docker compose logs prometheus")

    if trained_models:
        print("\n✅ Models are ready for forecasting!")
        print("\n🚀 Next step: Generate forecasts")
        print("   python3 /root/monitoring/scripts/forecast_metrics.py")
    else:
        print("\n❌ No models were trained successfully")
        print("   Please resolve the issues above and try again")
        sys.exit(1)

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
