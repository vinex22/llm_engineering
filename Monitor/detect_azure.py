#!/usr/bin/env python3
"""
AI-Powered Anomaly Detection - Azure Monitor Real-Time Detection Script

This script uses the trained IsolationForest model to detect anomalies
in real-time metrics from Azure Monitor.

Prerequisites:
    pip install azure-identity azure-mgmt-monitor pandas scikit-learn
    az login
"""

import os
import sys
import pickle
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.mgmt.monitor import MonitorManagementClient

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
# Replace these with your actual Azure resource details
SUBSCRIPTION_ID = "555a1e03-73fb-4f88-9296-59bd703d16f3"
RESOURCE_GROUP = "MC_aia-aks_aia-aks-private_eastus"
RESOURCE_NAME = "aks-systempool-90186695-vmss"
RESOURCE_PROVIDER = "Microsoft.Compute"
RESOURCE_TYPE = "virtualMachineScaleSets"
METRIC_NAME = "Network In Total"

MODEL_PATH = "azure_anomaly_model.pkl"

# Detection parameters
CHECK_INTERVAL = 60  # Check every 60 seconds (Azure metrics often have 1-min granularity)
LOOKBACK_MINUTES = 10  # Analyze last 10 minutes

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def load_model(path: str):
    """Load trained model from disk"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Please run train_azure.py first to train the model."
        )

    with open(path, 'rb') as f:
        model = pickle.load(f)

    print(f"   ✓ Model loaded from: {path}")
    return model

def fetch_recent_metrics(subscription_id, resource_group, resource_name, minutes):
    """
    Fetch recent metrics from Azure Monitor
    """
    credential = DefaultAzureCredential()
    client = MonitorManagementClient(credential, subscription_id)

    resource_id = (
        f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/"
        f"providers/{RESOURCE_PROVIDER}/{RESOURCE_TYPE}/{resource_name}"
    )

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=minutes)
    timespan = f"{start_time.isoformat()}/{end_time.isoformat()}"

    metrics_data = client.metrics.list(
        resource_id,
        timespan=timespan,
        interval='PT1M',  # 1-minute interval
        metricnames=METRIC_NAME,
        aggregation='Average'
    )

    timestamps = []
    values = []

    for item in metrics_data.value:
        for timeseries in item.timeseries:
            for data in timeseries.data:
                timestamps.append(data.time_stamp)
                values.append(data.average)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'metric_value': values
    })
    
    # Remove rows with None values (missing data)
    df = df.dropna()

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create same features as during training
    """
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Rolling statistics (last 5 samples)
    df['rolling_mean'] = df['metric_value'].rolling(window=5, min_periods=1).mean()
    df['rolling_std'] = df['metric_value'].rolling(window=5, min_periods=1).std().fillna(0)

    # Rate of change
    df['rate_of_change'] = df['metric_value'].diff().fillna(0)

    # Hour of day (for seasonality)
    df['hour'] = df['timestamp'].dt.hour

    # Drop NaN values
    df = df.dropna()

    return df

def detect_anomalies(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect anomalies using trained model
    """
    # Select same features used during training
    feature_columns = ['metric_value', 'rolling_mean', 'rolling_std', 'rate_of_change', 'hour']
    X = df[feature_columns]

    # Predict (-1 = anomaly, 1 = normal)
    predictions = model.predict(X)

    # Get anomaly scores (lower = more anomalous)
    scores = model.decision_function(X)

    # Add to DataFrame
    df['prediction'] = predictions
    df['anomaly_score'] = scores
    df['is_anomaly'] = predictions == -1

    return df

def print_detection_summary(df: pd.DataFrame):
    """Print summary of detection results"""
    total = len(df)
    anomalies = df['is_anomaly'].sum()
    normal = total - anomalies

    print(f"\n📊 Detection Summary (last {LOOKBACK_MINUTES} minutes):")
    print(f"   Total samples: {total}")
    print(f"   Normal: {normal} ({normal/total*100:.1f}%)")
    print(f"   Anomalies: {anomalies} ({anomalies/total*100:.1f}%)")

    if anomalies > 0:
        print(f"\n⚠️  {anomalies} ANOMALIES DETECTED!")
        print("\n🔍 Anomaly Details:")

        anomaly_df = df[df['is_anomaly']].copy()

        for idx, row in anomaly_df.iterrows():
            print(f"\n   Timestamp: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Metric Value: {row['metric_value']:.2f}")
            print(f"   Anomaly Score: {row['anomaly_score']:.4f} (lower = more anomalous)")
            print(f"   Rolling Mean: {row['rolling_mean']:.2f}")
            print(f"   Std Deviation: {row['rolling_std']:.2f}")
            print(f"   Rate of Change: {row['rate_of_change']:.2f}")

        # Analysis
        print(f"\n💡 Why these are anomalies:")

        high_val = anomaly_df[anomaly_df['metric_value'] > anomaly_df['metric_value'].quantile(0.75)]
        if len(high_val) > 0:
            print(f"   • {len(high_val)} samples: Unusually HIGH metric value")

        low_val = anomaly_df[anomaly_df['metric_value'] < anomaly_df['metric_value'].quantile(0.25)]
        if len(low_val) > 0:
            print(f"   • {len(low_val)} samples: Unusually LOW metric value")

        high_volatility = anomaly_df[anomaly_df['rolling_std'] > anomaly_df['rolling_std'].quantile(0.75)]
        if len(high_volatility) > 0:
            print(f"   • {len(high_volatility)} samples: Unusually HIGH volatility (rapid changes)")

    else:
        print(f"\n✅ No anomalies detected - all metrics within normal range")
        print("\n   The current behavior matches what the model learned")
        print("   as 'normal' during training:")
        
        if not df.empty:
            latest = df.iloc[-1]
            print(f"     - Current Value: {latest['metric_value']:.2f}")
            print(f"     - Rolling Average: {latest['rolling_mean']:.2f}")
            print(f"     - Std Deviation: {latest['rolling_std']:.2f}")

def continuous_monitoring(model, duration_minutes: int):
    """
    Run continuous anomaly detection for specified duration
    """
    print_header(f"🔄 Continuous Monitoring ({duration_minutes} minutes)")

    print(f"\n⏱️  Checking for anomalies every {CHECK_INTERVAL} seconds")
    print(f"📊 Analyzing last {LOOKBACK_MINUTES} minutes of data each check")
    print(f"⏰ Will run for {duration_minutes} minutes")
    print("\nPress Ctrl+C to stop early\n")

    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    check_count = 0

    try:
        while datetime.now() < end_time:
            check_count += 1
            current_time = datetime.now()

            print(f"\n{'─'*70}")
            print(f"Check #{check_count} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'─'*70}")

            # Fetch recent data
            df = fetch_recent_metrics(SUBSCRIPTION_ID, RESOURCE_GROUP, RESOURCE_NAME, LOOKBACK_MINUTES)

            if df.empty:
                print("   ⚠️ No data returned from Azure Monitor. Waiting for next check...")
            else:
                # Engineer features
                df = engineer_features(df)

                if df.empty:
                     print("   ⚠️ Not enough data points to engineer features. Waiting for next check...")
                else:
                    # Detect anomalies
                    df = detect_anomalies(model, df)

                    # Print results
                    print_detection_summary(df)

            # Wait before next check
            time_remaining = (end_time - datetime.now()).total_seconds()
            if time_remaining > CHECK_INTERVAL:
                print(f"\n⏳ Next check in {CHECK_INTERVAL} seconds...")
                time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n✅ Monitoring completed ({elapsed:.1f} minutes, {check_count} checks)")

def main():
    """Main detection workflow"""
    print_header("🤖 AI-Powered Anomaly Detection - Azure Monitor Real-Time Detection")

    print("\n🎯 This script will:")
    print("   1. Load the trained anomaly detection model")
    print("   2. Connect to Azure Monitor")
    print("   3. Run initial anomaly detection")
    print("   4. Automatically monitor for 5 minutes")
    print("   5. Ask if you want to continue monitoring")

    try:
        # Step 1: Load model
        print("\n[Step 1] Loading trained model")
        model = load_model(MODEL_PATH)

        # Step 2: Initial detection
        print("\n[Step 2] Running initial anomaly detection")
        df = fetch_recent_metrics(SUBSCRIPTION_ID, RESOURCE_GROUP, RESOURCE_NAME, LOOKBACK_MINUTES)
        print(f"   ✓ Fetched {len(df)} data points")

        if not df.empty:
            df = engineer_features(df)
            if not df.empty:
                df = detect_anomalies(model, df)
                print_detection_summary(df)
            else:
                print("   ⚠️ Not enough data points to engineer features.")
        else:
            print("   ⚠️ No data returned from Azure Monitor.")

        # Step 3: Automatic 5-minute continuous monitoring
        print("\n" + "="*70)
        print("\n🚀 Starting automatic 5-minute continuous monitoring...")
        continuous_monitoring(model, 5)

        # Step 4: Ask if user wants to continue monitoring
        while True:
            print("\n" + "="*70)
            choice = input("\n🔄 Continue monitoring? (y/n): ").strip().lower()

            if choice == 'y':
                try:
                    duration = int(input("⏱️  How many more minutes to monitor? (e.g., 5): ").strip())
                    if duration > 0:
                        continuous_monitoring(model, duration)
                    else:
                        print("\n⚠️  Duration must be greater than 0. Stopping monitoring.")
                        break
                except ValueError:
                    print("\n⚠️  Invalid input. Please enter a number. Stopping monitoring.")
                    break
            else:
                print("\n✅ Monitoring complete!")
                break

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n🔧 Solution:")
        print("   Run the training script first:")
        print("   python Monitor/train_azure.py")
        return 1

    except Exception as e:
        print(f"\n❌ Error during detection: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
