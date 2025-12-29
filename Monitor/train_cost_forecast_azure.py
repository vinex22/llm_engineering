#!/usr/bin/env python3
"""
AI-Driven Forecasting - Azure Cost Prediction
Module 5: AIOps Foundations

This script fetches daily cost data from Azure Cost Management and trains
a Prophet model to forecast future cloud spending.

Prerequisites:
    pip install azure-identity azure-mgmt-costmanagement pandas prophet
    az login
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

# Aggressively suppress Prophet/Stan logging
for logger_name in ['prophet', 'cmdstanpy']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True

logging.getLogger('azure').setLevel(logging.ERROR)

try:
    import pandas as pd
    from prophet import Prophet
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.costmanagement import CostManagementClient
    from azure.mgmt.costmanagement.models import (
        QueryDefinition,
        QueryTimePeriod,
        QueryDataset,
        QueryAggregation,
        QueryGrouping
    )
except ImportError as e:
    print(f"❌ Error: Missing required package: {e}")
    print("\n💡 Please install packages:")
    print("   pip install azure-identity azure-mgmt-costmanagement pandas prophet")
    sys.exit(1)

# Configuration
SUBSCRIPTION_ID = "555a1e03-73fb-4f88-9296-59bd703d16f3"  # Matches your other scripts
TRAINING_DAYS = 90  # Look back 90 days
FORECAST_DAYS = 90  # Predict next 30 days

def print_header():
    print("\n" + "="*70)
    print("💰 Azure Cost Forecasting with Prophet")
    print("Module 5: AIOps Foundations")
    print("="*70 + "\n")

def get_cost_data(subscription_id, days=90):
    """Fetch daily cost data from Azure, grouped by Meter."""
    print(f"🔌 Connecting to Azure Subscription: {subscription_id}...")
    
    try:
        credential = DefaultAzureCredential()
        client = CostManagementClient(credential)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"📊 Fetching cost data from {start_date.date()} to {end_date.date()}...")

        # Define the query for daily costs grouped by Meter
        query = QueryDefinition(
            type="Usage",
            timeframe="Custom",
            time_period=QueryTimePeriod(
                from_property=start_date,
                to=end_date
            ),
            dataset=QueryDataset(
                granularity="Daily",
                aggregation={
                    "totalCost": QueryAggregation(name="Cost", function="Sum")
                },
                grouping=[
                    QueryGrouping(type="Dimension", name="Meter")
                ]
            )
        )

        # Execute query
        scope = f"/subscriptions/{subscription_id}"
        result = client.query.usage(scope, query)
        
        # Convert to DataFrame
        columns = [col.name for col in result.columns]
        rows = result.rows
        
        if not rows:
            print("⚠️  No cost data found for this period.")
            return None
            
        df = pd.DataFrame(rows, columns=columns)
        
        # Normalize columns
        date_col = next((c for c in columns if 'Date' in c), 'UsageDate')
        cost_col = next((c for c in columns if 'Cost' in c), 'totalCost')
        # Map 'Meter' column to 'service' for compatibility with existing functions
        service_col = next((c for c in columns if 'Meter' in c), 'Meter')
        
        df = df.rename(columns={date_col: 'date', cost_col: 'cost', service_col: 'service'})
        
        # Ensure date is datetime
        if pd.api.types.is_numeric_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        print(f"✅ Retrieved {len(df)} records")
        print(f"   Total Cost in period: ${df['cost'].sum():.2f}")
        
        return df

    except Exception as e:
        print(f"❌ Error fetching Azure cost data: {e}")
        return None

def prepare_prophet_data(df, service_name=None):
    """Filter data by service (optional) and prepare for Prophet."""
    if service_name and service_name != 'Total':
        # Filter for specific service
        df_filtered = df[df['service'] == service_name].copy()
    else:
        # Group by date to get total daily cost
        df_filtered = df.groupby('date')['cost'].sum().reset_index()
        
    # Rename for Prophet
    # If we filtered by service, we still have 'date' and 'cost' columns
    # If we grouped by date, we have 'date' and 'cost' columns
    df_prophet = df_filtered.rename(columns={'date': 'ds', 'cost': 'y'})
    
    return df_prophet

def select_service(df):
    """Deprecated: Interactive meter selection."""
    pass

def train_and_forecast(df, periods=90):
    """Train Prophet model and generate forecast."""
    # Suppress Prophet output
    logging.getLogger('prophet').setLevel(logging.CRITICAL)
    logging.getLogger('cmdstanpy').setLevel(logging.CRITICAL)
    logging.getLogger('cmdstanpy').disabled = True
    
    try:
        # Initialize Prophet
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Predict
        forecast = model.predict(future)
        
        # Clamp negative values to 0
        cols_to_clamp = ['yhat', 'yhat_lower', 'yhat_upper']
        for col in cols_to_clamp:
            if col in forecast.columns:
                forecast[col] = forecast[col].clip(lower=0)
        
        return model, forecast
        
    except Exception as e:
        # print(f"❌ Error training model: {e}")
        return None, None

def print_report_row(service_name, df, forecast, status="Active"):
    """Print a single row summary for a service with actuals vs forecast."""
    
    last_actual_date = df['ds'].max()
    
    # Calculate Actuals (Last 30 days)
    start_date_30d_ago = last_actual_date - timedelta(days=30)
    actual_30 = df[df['ds'] > start_date_30d_ago]['y'].sum()
    
    # Calculate Average Daily Cost (Last 7 days) - for sanity check
    start_date_7d_ago = last_actual_date - timedelta(days=7)
    avg_7d = df[df['ds'] > start_date_7d_ago]['y'].mean()
    if pd.isna(avg_7d): avg_7d = 0.0

    # Calculate forecasts for different horizons
    future_forecast = forecast[forecast['ds'] > last_actual_date]
    
    def get_forecast_sum(days):
        target_date = last_actual_date + timedelta(days=days)
        mask = (future_forecast['ds'] <= target_date)
        return future_forecast.loc[mask, 'yhat'].sum()

    cost_30 = get_forecast_sum(30)
    cost_60 = get_forecast_sum(60)
    cost_90 = get_forecast_sum(90)
    
    # Trend
    avg_daily_hist = df['y'].mean()
    avg_daily_future = future_forecast['yhat'].mean()
    trend = "↗" if avg_daily_future > avg_daily_hist else "↘"
    
    # Status indicator
    status_icon = ""
    if status == "New": status_icon = "🆕"
    elif status == "Inactive": status_icon = "❌"
    
    # Print with Actuals column
    name_display = f"{status_icon} {service_name[:23]}"
    print(f"{name_display:<27} ${avg_7d:<9.2f} ${actual_30:<10.2f} ${cost_30:<10.2f} ${cost_60:<10.2f} ${cost_90:<10.2f} {trend}")

def generate_flat_forecast(df, periods=90):
    """Generate a flat forecast based on recent average (for new services)."""
    last_date = df['ds'].max()
    
    # Use average of last 3 days
    start_avg = last_date - timedelta(days=3)
    daily_avg = df[df['ds'] > start_avg]['y'].mean()
    if pd.isna(daily_avg): daily_avg = df.iloc[-1]['y']
    
    future_dates = [last_date + timedelta(days=x) for x in range(1, periods + 1)]
    
    forecast = pd.DataFrame({
        'ds': future_dates,
        'yhat': [daily_avg] * periods,
        'yhat_lower': [daily_avg] * periods,
        'yhat_upper': [daily_avg] * periods
    })
    
    # Combine with history for compatibility
    history = df[['ds', 'y']].rename(columns={'y': 'yhat'})
    history['yhat_lower'] = history['yhat']
    history['yhat_upper'] = history['yhat']
    
    full_forecast = pd.concat([history, forecast])
    return None, full_forecast

def main():
    print_header()
    
    # 1. Get Data
    df_raw = get_cost_data(SUBSCRIPTION_ID, days=TRAINING_DAYS)
    if df_raw is None or df_raw.empty:
        return

    # Global max date to detect inactive services
    global_max_date = df_raw['date'].max()
    print(f"📅 Data available up to: {global_max_date.date()}")

    # Get all unique meters
    meters = df_raw['service'].unique()
    
    print("\n" + "="*100)
    print(f"🔮 Forecast Summary (Actuals vs Forecast)")
    print("="*100)
    print(f"{'Meter Name':<27} {'Avg(7d)':<10} {'Act(30d)':<11} {'Fcst(30d)':<11} {'Fcst(60d)':<11} {'Fcst(90d)':<11} {'Trend'}")
    print("-" * 100)

    # Process each meter
    total_act_30 = 0
    total_30 = 0
    total_60 = 0
    total_90 = 0
    
    # Sort meters by total historical cost
    meter_costs = df_raw.groupby('service')['cost'].sum().sort_values(ascending=False)
    
    for service_name in meter_costs.index:
        # Prepare data
        df_prophet = prepare_prophet_data(df_raw, service_name)
        
        if len(df_prophet) < 1:
            continue
            
        # Determine Status
        last_date = df_prophet['ds'].max()
        first_date = df_prophet['ds'].min()
        
        days_inactive = (global_max_date - last_date).days
        age_days = (last_date - first_date).days
        
        status = "Active"
        model = None
        forecast = None
        
        if days_inactive > 2:
            status = "Inactive"
            # Forecast is 0 for inactive services
            future_dates = [last_date + timedelta(days=x) for x in range(1, 91)]
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': [0] * 90
            })
            # Add history
            history = df_prophet[['ds', 'y']].rename(columns={'y': 'yhat'})
            forecast = pd.concat([history, forecast])
            
        elif age_days < 14:
            status = "New"
            # Use flat forecast for new services to avoid wild extrapolation
            model, forecast = generate_flat_forecast(df_prophet, periods=90)
            
        else:
            # Standard Prophet forecast
            if len(df_prophet) >= 5:
                model, forecast = train_and_forecast(df_prophet, periods=90)
            else:
                # Not enough data for Prophet but active -> Flat forecast
                model, forecast = generate_flat_forecast(df_prophet, periods=90)
        
        if forecast is not None:
            print_report_row(service_name, df_prophet, forecast, status)
            
            # Add to totals
            future = forecast[forecast['ds'] > last_date]
            
            # Actuals total
            start_date_30d_ago = last_date - timedelta(days=30)
            total_act_30 += df_prophet[df_prophet['ds'] > start_date_30d_ago]['y'].sum()
            
            # Forecast totals
            # Note: We align forecast to the global timeline for totals
            # If a service is inactive, its forecast is 0, so it adds 0.
            # If a service is active, its forecast starts from its last_date.
            
            # We need to be careful summing up. 
            # We want the sum of costs for the NEXT 30 days from TODAY (global_max_date).
            
            # Filter forecast for dates > global_max_date
            future_global = forecast[forecast['ds'] > global_max_date]
            
            total_30 += future_global[future_global['ds'] <= global_max_date + timedelta(days=30)]['yhat'].sum()
            total_60 += future_global[future_global['ds'] <= global_max_date + timedelta(days=60)]['yhat'].sum()
            total_90 += future_global[future_global['ds'] <= global_max_date + timedelta(days=90)]['yhat'].sum()

    print("-" * 100)
    print(f"{'TOTAL':<27} {'-':<10} ${total_act_30:<10.2f} ${total_30:<10.2f} ${total_60:<10.2f} ${total_90:<10.2f}")
    print("="*100 + "\n")
    print("Legend: 🆕 = New Service (<14 days), ❌ = Inactive/Deleted (>2 days no data)")

if __name__ == "__main__":
    main()
