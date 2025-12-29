#!/usr/bin/env python3
"""
AI-Powered Anomaly Detection - Azure Monitor Training Script

This script trains an IsolationForest model on metrics fetched from Azure Monitor.
It learns what "normal" behavior looks like and saves the model for later use.

Prerequisites:
    pip install azure-identity azure-mgmt-monitor pandas scikit-learn
    az login
"""

import os
import sys
import pickle
import warnings
from datetime import datetime, timedelta

import pandas as pd
from sklearn.ensemble import IsolationForest
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

# Training parameters
TRAINING_HOURS = 30 * 24  # Use last 30 days of data for training
CONTAMINATION = 0.1  # Expected proportion of outliers (10%)
MIN_SAMPLES_REQUIRED = 15  # Minimum samples needed for training

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_step(step_num, text):
    """Print a formatted step"""
    print(f"\n[Step {step_num}] {text}")

def fetch_azure_metrics(subscription_id, resource_group, resource_name, hours):
    """
    Fetch metrics from Azure Monitor
    """
    credential = DefaultAzureCredential()
    client = MonitorManagementClient(credential, subscription_id)

    resource_id = (
        f"/subscriptions/{subscription_id}/resourceGroups/{resource_group}/"
        f"providers/{RESOURCE_PROVIDER}/{RESOURCE_TYPE}/{resource_name}"
    )

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    timespan = f"{start_time.isoformat()}/{end_time.isoformat()}"

    print(f"   Querying Azure Monitor from {start_time} to {end_time}")

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

    print(f"   ✓ Fetched {len(df)} data points")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features for machine learning
    """
    print("   Creating features:")
    print("     - Rolling mean (5 samples)")
    print("     - Rolling std (5 samples)")
    print("     - Rate of change")

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Rolling statistics
    df['rolling_mean'] = df['metric_value'].rolling(window=5, min_periods=1).mean()
    df['rolling_std'] = df['metric_value'].rolling(window=5, min_periods=1).std().fillna(0)

    # Rate of change
    df['rate_of_change'] = df['metric_value'].diff().fillna(0)

    # Hour of day (for seasonality)
    df['hour'] = df['timestamp'].dt.hour

    # Drop NaN values
    df = df.dropna()

    print(f"   ✓ Created {len(df.columns) - 2} features")  # -2 for timestamp and metric_value

    return df

def train_model(df: pd.DataFrame, contamination: float) -> IsolationForest:
    """
    Train IsolationForest model
    """
    # Select features for training
    feature_columns = ['metric_value', 'rolling_mean', 'rolling_std', 'rate_of_change', 'hour']
    X = df[feature_columns]

    print(f"   Training on {len(X)} samples with {len(feature_columns)} features")
    print(f"   Contamination: {contamination*100}% (expected outlier proportion)")

    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        verbose=0
    )

    model.fit(X)

    # Calculate training statistics
    predictions = model.predict(X)
    anomalies = (predictions == -1).sum()
    normal = (predictions == 1).sum()

    print(f"   ✓ Model trained successfully")
    print(f"   Training results:")
    print(f"     - Normal samples: {normal} ({normal/len(X)*100:.1f}%)")
    print(f"     - Anomalies detected: {anomalies} ({anomalies/len(X)*100:.1f}%)")

    return model

def save_model(model: IsolationForest, path: str):
    """Save trained model to disk"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    print(f"   ✓ Model saved to: {path}")
    print(f"   Model size: {os.path.getsize(path) / 1024:.2f} KB")

def main():
    """Main training workflow"""
    print_header("🤖 AI-Powered Anomaly Detection - Azure Monitor Training")

    # Check environment variables
    # if not all([SUBSCRIPTION_ID, RESOURCE_GROUP, RESOURCE_NAME]):
    #     print("\n❌ Error: Missing environment variables.")
    #     print("   Please set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, and AZURE_RESOURCE_NAME.")
    #     return 1

    try:
        # Step 1: Fetch metrics
        print_step(1, f"Fetching up to {TRAINING_HOURS} hour(s) of metrics from Azure Monitor")
        df = fetch_azure_metrics(SUBSCRIPTION_ID, RESOURCE_GROUP, RESOURCE_NAME, TRAINING_HOURS)

        if len(df) < MIN_SAMPLES_REQUIRED:
            print(f"\n❌ Error: Only {len(df)} data points available.")
            print(f"   Minimum required: {MIN_SAMPLES_REQUIRED} samples")
            return 1

        # Step 2: Feature engineering
        print_step(2, "Engineering features for ML")
        df = engineer_features(df)

        # Step 3: Train model
        print_step(3, "Training IsolationForest model")
        model = train_model(df, CONTAMINATION)

        # Step 4: Save model
        print_step(4, "Saving trained model")
        save_model(model, MODEL_PATH)

        print_header("✅ Training Complete!")
        return 0

    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
