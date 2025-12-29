#!/usr/bin/env python3
"""
AI-Driven Forecasting - Prediction Script
Module 5: AIOps Foundations

This script uses trained Prophet models to generate forecasts for infrastructure
metrics and predict capacity exhaustion dates for proactive planning.
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

# Metrics configuration (must match train_forecasting_model.py)
METRICS_CONFIG = {
    'cpu_usage': {
        'name': 'CPU Usage',
        'unit': '%',
        'threshold': 80,
        'threshold_type': 'upper',  # Alert when forecast exceeds threshold
        'format': '.1f'
    },
    'memory_available': {
        'name': 'Memory Available',
        'unit': 'GB',
        'threshold': 3.0,
        'threshold_type': 'lower',  # Alert when forecast drops below threshold
        'format': '.2f'
    },
    'disk_usage': {
        'name': 'Disk Usage',
        'unit': '%',
        'threshold': 90,
        'threshold_type': 'upper',
        'format': '.1f'
    }
}


def print_header():
    """Print script header."""
    print("\n" + "="*70)
    print("📈 AI-Driven Forecasting - Generate Predictions")
    print("Module 5: AIOps Foundations")
    print("="*70 + "\n")


def connect_to_prometheus():
    """Connect to Prometheus API."""
    try:
        prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
        prom.check_prometheus_connection()
        return prom
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to Prometheus: {e}")
        return None


def get_current_value(prom, metric_key):
    """Fetch current metric value from Prometheus."""
    if prom is None:
        return None

    try:
        # Get the query for this metric from training config
        query = None
        if metric_key == 'cpu_usage':
            query = '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
        elif metric_key == 'memory_available':
            query = 'node_memory_MemAvailable_bytes / 1024 / 1024 / 1024'
        elif metric_key == 'disk_usage':
            query = '(1 - node_filesystem_avail_bytes{fstype=~"ext4|xfs|btrfs"} / node_filesystem_size_bytes{fstype=~"ext4|xfs|btrfs"}) * 100'

        if query is None:
            return None

        # Query current value
        result = prom.custom_query(query=query)

        if result and len(result) > 0:
            value = float(result[0]['value'][1])
            return value
        else:
            return None
    except Exception as e:
        print(f"⚠️  Warning: Could not fetch current value for {metric_key}: {e}")
        return None


def load_model(metric_key):
    """Load trained model from disk."""
    model_path = os.path.join(MODEL_DIR, f"{metric_key}_forecast_model.pkl")

    if not os.path.exists(model_path):
        return None

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"❌ Error loading model for {metric_key}: {e}")
        return None


def generate_forecast(model, metric_key, periods=7):
    """Generate forecast for specified number of days."""
    try:
        # Create future dataframe - use 30 second intervals to match training data
        # periods in this case is number of 30-second intervals for N days
        future_periods = periods * 24 * 60 * 2  # days * hours * minutes * (60s/30s)
        future = model.make_future_dataframe(periods=future_periods, freq='30S')

        # For logistic growth models, need to add cap and floor
        # Check if model has logistic growth by examining model.growth
        if hasattr(model, 'growth') and model.growth == 'logistic':
            future['cap'] = 100.0  # For percentage metrics

            # Set realistic floor based on metric type
            if metric_key == 'cpu_usage':
                future['floor'] = 1.0   # CPU can't be 0% on running system
            else:
                future['floor'] = 0.0   # Disk can theoretically be 0%

        # Generate forecast
        forecast = model.predict(future)

        return forecast

    except AttributeError as e:
        # Handle Prophet 1.1.5 stan_backend AttributeError
        if 'stan_backend' in str(e):
            print(f"   Warning: Prophet stan_backend deprecation issue")
            # Try to continue anyway
            return None
        else:
            print(f"❌ Error generating forecast: {e}")
            import traceback
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"❌ Error generating forecast: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_forecast(forecast, config, current_value, periods=7):
    """Analyze forecast results and identify capacity issues."""
    # Calculate number of future data points (at 30-second intervals)
    future_points = periods * 24 * 60 * 2  # days * hours * minutes * (60s/30s)

    # Get only the FUTURE forecast period (exclude historical data)
    forecast_period = forecast.tail(future_points)

    # Extract key metrics from forecast period
    forecast_mean = forecast_period['yhat'].mean()
    forecast_max = forecast_period['yhat'].max()
    forecast_min = forecast_period['yhat'].min()

    # Calculate trend
    trend_start = forecast_period['yhat'].iloc[0]
    trend_end = forecast_period['yhat'].iloc[-1]
    trend_change = trend_end - trend_start
    trend_per_day = trend_change / periods

    # Calculate confidence interval width
    ci_width = (forecast_period['yhat_upper'] - forecast_period['yhat_lower']).mean()

    # Check for threshold breach
    threshold = config['threshold']
    threshold_type = config['threshold_type']

    days_to_threshold = None
    threshold_breached = False

    if threshold_type == 'upper':
        # Check if forecast exceeds upper threshold
        breach_df = forecast_period[forecast_period['yhat'] > threshold]
        if not breach_df.empty:
            threshold_breached = True
            # Calculate days until first breach (using row position, not index)
            first_breach_pos = breach_df.index.get_loc(breach_df.index[0])
            forecast_start_pos = forecast_period.index.get_loc(forecast_period.index[0])
            points_to_breach = first_breach_pos - forecast_start_pos
            # Convert points (30s intervals) to days
            days_to_threshold = (points_to_breach * 30) / (24 * 60 * 60)

    elif threshold_type == 'lower':
        # Check if forecast drops below lower threshold
        breach_df = forecast_period[forecast_period['yhat'] < threshold]
        if not breach_df.empty:
            threshold_breached = True
            first_breach_pos = breach_df.index.get_loc(breach_df.index[0])
            forecast_start_pos = forecast_period.index.get_loc(forecast_period.index[0])
            points_to_breach = first_breach_pos - forecast_start_pos
            # Convert points (30s intervals) to days
            days_to_threshold = (points_to_breach * 30) / (24 * 60 * 60)

    return {
        'current_value': current_value,
        'forecast_mean': forecast_mean,
        'forecast_max': forecast_max,
        'forecast_min': forecast_min,
        'trend_per_day': trend_per_day,
        'ci_width': ci_width,
        'threshold_breached': threshold_breached,
        'days_to_threshold': days_to_threshold
    }


def print_forecast_report(metric_key, config, analysis, periods):
    """Print clean, compact forecast report."""
    fmt = config['format']
    unit = config['unit']

    print(f"\n{config['name']} Forecast:")
    print(f"  Current: {analysis['current_value']:{fmt}}{unit}")

    # {periods}-day prediction with range
    forecast_range = f"{analysis['forecast_min']:{fmt}}{unit} - {analysis['forecast_max']:{fmt}}{unit}"
    print(f"  {periods}-day prediction: {analysis['forecast_mean']:{fmt}}{unit} (range: {forecast_range})")

    # Trend analysis with arrow
    trend = analysis['trend_per_day']
    if abs(trend) > 0.01:
        arrow = "↗" if trend > 0 else "↘"
        direction = "Increasing" if trend > 0 else "Decreasing"
        total_change = trend * periods
        print(f"  Trend: {arrow} {direction} ({total_change:+{fmt}}{unit} over {periods} days, {trend:+{fmt}}{unit}/day)")
    else:
        print(f"  Trend: → Stable (minimal change)")

    # Threshold status
    threshold = config['threshold']
    threshold_type = config['threshold_type']

    if analysis['threshold_breached']:
        days = analysis['days_to_threshold']

        if threshold_type == 'upper':
            verb = "exceed"
            threshold_msg = f"will exceed {threshold}{unit}"
        else:
            verb = "drop below"
            threshold_msg = f"will drop below {threshold}{unit}"

        # Urgency indicator
        if days < 7:
            urgency = "🚨"
        elif days < 14:
            urgency = "⚠️"
        else:
            urgency = "💡"

        print(f"  Alert: {urgency} {config['name']} {threshold_msg} in ~{days:.1f} days")

        # Single-line recommendation
        if metric_key == 'disk_usage':
            recommendation = "Plan storage expansion or cleanup"
        elif metric_key == 'cpu_usage':
            recommendation = "Consider scaling up or optimizing workloads"
        elif metric_key == 'memory_available':
            recommendation = "Investigate memory leaks or add more RAM"
        else:
            recommendation = "Review capacity planning"

        print(f"  Recommendation: {recommendation}")
    else:
        if threshold_type == 'upper':
            status_msg = f"will not exceed {threshold}{unit}"
        else:
            status_msg = f"will not drop below {threshold}{unit}"

        print(f"  Status: ✅ {config['name']} {status_msg} within {periods} days")
        print(f"  Recommendation: Monitor trend, no immediate action needed")


def prompt_forecast_horizon():
    """Ask user for forecast horizon."""
    print("📅 Select forecast horizon:")
    print("   1) 7 days (recommended for short-term planning)")
    print("   2) 14 days (recommended for capacity orders)")
    print("   3) 30 days (recommended for quarterly planning)")
    print("   4) Custom")

    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '1':
                return 7
            elif choice == '2':
                return 14
            elif choice == '3':
                return 30
            elif choice == '4':
                days = input("Enter number of days (1-90): ").strip()
                days = int(days)
                if 1 <= days <= 90:
                    return days
                else:
                    print("❌ Please enter a value between 1 and 90")
            else:
                print("❌ Invalid choice, please enter 1-4")

        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input, using default: 7 days")
            return 7


def main():
    """Main forecasting workflow."""
    print_header()

    # Check if model directory exists
    if not os.path.exists(MODEL_DIR):
        print(f"❌ Model directory not found: {MODEL_DIR}")
        print("\n💡 Please train models first:")
        print("   python3 /root/monitoring/scripts/train_forecasting_model.py")
        sys.exit(1)

    # Load all available models
    print("🔍 Loading trained models...")
    models = {}
    for metric_key, config in METRICS_CONFIG.items():
        model = load_model(metric_key)
        if model:
            models[metric_key] = model
            print(f"   ✅ Loaded model for {config['name']}")
        else:
            print(f"   ⚠️  Model not found for {config['name']}")

    if not models:
        print("\n❌ No trained models found!")
        print("💡 Please train models first:")
        print("   python3 /root/monitoring/scripts/train_forecasting_model.py")
        sys.exit(1)

    print(f"\n✅ Loaded {len(models)} model(s)\n")

    # Connect to Prometheus to fetch current values
    print("🔌 Connecting to Prometheus...")
    prom = connect_to_prometheus()
    if prom:
        print("✅ Connected to Prometheus\n")
    else:
        print("⚠️  Could not connect to Prometheus - will use training end values\n")

    # Get forecast horizon from user
    periods = prompt_forecast_horizon()

    print(f"\n🔮 Generating {periods}-day forecasts...\n")

    # Generate forecasts for each metric
    forecast_results = {}

    for metric_key, model in models.items():
        config = METRICS_CONFIG[metric_key]

        print(f"📊 Processing {config['name']}...")

        # Get current value from Prometheus
        current_value = get_current_value(prom, metric_key)
        if current_value is None:
            print(f"   ⚠️  Could not fetch current value, using training end value")
            # Fallback to Prophet's fitted value from training end
            # This requires generating forecast first to access historical data
            temp_forecast = generate_forecast(model, metric_key, periods=1)
            if temp_forecast is not None:
                future_points = 1 * 24 * 60 * 2  # 1 day in 30-second intervals
                current_value = temp_forecast.iloc[-(future_points + 1)]['yhat']
            else:
                print(f"   ❌ Failed to get current value for {config['name']}")
                continue

        # Generate forecast
        forecast = generate_forecast(model, metric_key, periods=periods)

        if forecast is None:
            print(f"   ❌ Failed to generate forecast for {config['name']}")
            continue

        # Analyze forecast
        analysis = analyze_forecast(forecast, config, current_value, periods=periods)

        # Store results
        forecast_results[metric_key] = {
            'forecast': forecast,
            'analysis': analysis
        }

        print(f"   ✅ Forecast generated for {config['name']}")

    # Print detailed reports
    print("\n" + "="*70)
    print(f"📊 {periods}-Day Forecast Results")
    print("="*70)

    for metric_key, results in forecast_results.items():
        config = METRICS_CONFIG[metric_key]
        print_forecast_report(
            metric_key,
            config,
            results['analysis'],
            periods
        )

    # Print summary
    print("\n" + "="*70)
    print("📋 Summary")
    print("="*70)

    alerts = []
    ok_metrics = []

    for metric_key, results in forecast_results.items():
        config = METRICS_CONFIG[metric_key]
        analysis = results['analysis']

        if analysis['threshold_breached']:
            alerts.append({
                'metric': config['name'],
                'days': analysis['days_to_threshold']
            })
        else:
            ok_metrics.append(config['name'])

    if alerts:
        print(f"\n⚠️  Capacity Alerts ({len(alerts)}):")
        # Sort by urgency (days to threshold)
        alerts.sort(key=lambda x: x['days'])
        for alert in alerts:
            urgency = "🚨" if alert['days'] < 7 else "⚠️" if alert['days'] < 14 else "💡"
            print(f"  {urgency} {alert['metric']}: Action needed in ~{alert['days']:.1f} days")

    if ok_metrics:
        print(f"\n✅ Healthy Metrics ({len(ok_metrics)}):")
        for metric in ok_metrics:
            print(f"  • {metric}")

    print("\n" + "="*70)
    print("✨ Forecast generation complete!")
    print("\n💡 Next Steps:")
    print("  • Review forecasts weekly and retrain models monthly")
    print("  • Compare predictions to actual values to validate accuracy")
    print("  • View detailed metrics in Grafana dashboard")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
