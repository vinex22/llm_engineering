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
FORECAST_DAYS = 365  # Predict next 12 months

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

def train_and_forecast(df, periods=365):
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
        
        # Add Hong Kong holidays
        try:
            model.add_country_holidays(country_name='HK')
        except Exception:
            # Fallback if holidays package not installed or country code invalid
            pass
        
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

def print_report_row(service_name, df, forecast, months, status="Active"):
    """Print a single row summary with monthly breakdown."""
    
    # Calculate forecasts for the specific months
    forecast = forecast.copy()
    forecast['month'] = forecast['ds'].dt.to_period('M')
    monthly_sums = forecast.groupby('month')['yhat'].sum()
    
    # Status indicator
    status_icon = ""
    if status == "New": status_icon = "🆕"
    elif status == "Inactive": status_icon = "❌"
    
    # Prepare row string
    name_display = f"{status_icon} {service_name[:20]}" # Shorten name to fit
    row_str = f"{name_display:<24}"
    
    total_row_cost = 0
    for m in months:
        val = monthly_sums.get(m, 0.0)
        row_str += f" ${val:<8.0f}" 
        total_row_cost += val
        
    row_str += f" ${total_row_cost:<9.0f}"
    print(row_str)

def generate_flat_forecast(df, periods=400):
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

    # Global max date
    global_max_date = df_raw['date'].max()
    print(f"📅 Data available up to: {global_max_date.date()}")

    # Define the 12 months to display (starting next full month)
    # If we are in Dec 29, next full month is Jan.
    start_date = global_max_date + timedelta(days=1)
    start_period = pd.Period(start_date, freq='M')
    
    # If we are late in the month (>20th), start next month to avoid partial month confusion
    if start_date.day > 20:
        start_period = start_period + 1
        
    months = [start_period + i for i in range(12)]
    
    # Print Headers
    header_str = f"{'Meter Name':<24}"
    for m in months:
        month_label = m.strftime("%b'%y")
        header_str += f" {month_label:<9}"
    header_str += f" {'Total':<10}"
    
    print("\n" + "="*len(header_str))
    print(f"🔮 12-Month Forecast Breakdown")
    print("="*len(header_str))
    print(header_str)
    print("-" * len(header_str))

    # Process each meter
    totals = {m: 0.0 for m in months}
    grand_total = 0
    
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
        
        # Forecast 400 days to ensure we cover the 12 months
        forecast_days = 400
        
        if days_inactive > 2:
            status = "Inactive"
            future_dates = [last_date + timedelta(days=x) for x in range(1, forecast_days + 1)]
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': [0] * forecast_days
            })
            history = df_prophet[['ds', 'y']].rename(columns={'y': 'yhat'})
            forecast = pd.concat([history, forecast])
            
        elif age_days < 14:
            status = "New"
            model, forecast = generate_flat_forecast(df_prophet, periods=forecast_days)
            
        else:
            if len(df_prophet) >= 5:
                model, forecast = train_and_forecast(df_prophet, periods=forecast_days)
            else:
                model, forecast = generate_flat_forecast(df_prophet, periods=forecast_days)
        
        if forecast is not None:
            print_report_row(service_name, df_prophet, forecast, months, status)
            
            # Add to totals
            forecast['month'] = forecast['ds'].dt.to_period('M')
            monthly_sums = forecast.groupby('month')['yhat'].sum()
            
            row_total = 0
            for m in months:
                val = monthly_sums.get(m, 0.0)
                totals[m] += val
                row_total += val
            grand_total += row_total

    print("-" * len(header_str))
    total_str = f"{'TOTAL':<24}"
    for m in months:
        total_str += f" ${totals[m]:<8.0f}"
    total_str += f" ${grand_total:<9.0f}"
    print(total_str)
    print("="*len(header_str) + "\n")
    print("Legend: 🆕 = New Service (<14 days), ❌ = Inactive/Deleted (>2 days no data)")

if __name__ == "__main__":
    main()
