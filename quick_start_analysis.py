#!/usr/bin/env python3
"""
OCP Predictive Maintenance - Quick Start Analysis Script
========================================================
This script performs initial data loading and exploration for the hackathon.

Author: Hackathon Team
Date: 2024
"""

import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Configuration
DATA_FILE = "Dataframemin.csv"
FAILURE_LOG = "Maintenance prédictive/Suivi des arrêts_ligne 307_2019.xlsx"
SKIP_ROWS = 15609  # Skip rows with "No Data"

# Sensor mapping
SENSOR_MAPPING = {
    "307II546": "Motor_Current_A",
    "307TI178A": "Temp_Opposite_C",
    "307TI178B": "Temp_Motor_C",
    "307VI746A": "Vibration_Opposite_mm_s",
    "307VI746B": "Vibration_Motor_mm_s",
    "307ZI717": "Valve_Opening_Percent",
}


def load_sensor_data(filepath=DATA_FILE, skip_rows=SKIP_ROWS):
    """
    Load sensor data from CSV, skipping incomplete rows.

    Returns:
        pd.DataFrame: Clean sensor data with datetime index
    """
    print("=" * 70)
    print("STEP 1: LOADING SENSOR DATA")
    print("=" * 70)

    print(f"\n📂 Loading data from: {filepath}")
    print(f"⏭️  Skipping first {skip_rows} rows (incomplete data)")

    # Load data, skipping rows with "No Data"
    df = pd.read_csv(filepath, skiprows=range(1, skip_rows + 1))

    print(f"✅ Loaded {len(df):,} rows")
    print(f"\nColumns: {df.columns.tolist()}")

    # Parse dates
    print("\n🕐 Parsing timestamps...")
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")

    # Rename columns for clarity
    df = df.rename(columns=SENSOR_MAPPING)

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    # Basic info
    print(f"📅 Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"⏱️  Duration: {(df['Date'].max() - df['Date'].min()).days} days")
    print(f"📊 Data points: {len(df):,} (minute-by-minute)")

    return df


def data_quality_check(df):
    """
    Check data quality and identify issues.
    """
    print("\n" + "=" * 70)
    print("STEP 2: DATA QUALITY CHECK")
    print("=" * 70)

    sensor_cols = [col for col in df.columns if col != "Date"]

    print("\n📊 Missing Values:")
    missing = df[sensor_cols].isnull().sum()
    for col, count in missing.items():
        pct = (count / len(df)) * 100
        print(f"  {col}: {count} ({pct:.2f}%)")

    print("\n📈 Basic Statistics:")
    print(df[sensor_cols].describe())

    print("\n🔍 Outlier Detection (values beyond 3 std dev):")
    for col in sensor_cols:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)]
        if len(outliers) > 0:
            print(
                f"  {col}: {len(outliers)} outliers ({len(outliers) / len(df) * 100:.2f}%)"
            )

    return df


def explore_sensor_patterns(df):
    """
    Explore patterns in sensor data.
    """
    print("\n" + "=" * 70)
    print("STEP 3: SENSOR PATTERN EXPLORATION")
    print("=" * 70)

    sensor_cols = [col for col in df.columns if col != "Date"]

    print("\n📊 Sensor Value Ranges:")
    for col in sensor_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"\n  {col}:")
        print(f"    Range: [{min_val:.2f}, {max_val:.2f}]")
        print(f"    Mean: {mean_val:.2f} ± {std_val:.2f}")

    print("\n🔗 Correlation Analysis:")
    corr_matrix = df[sensor_cols].corr()
    print(corr_matrix)

    # Identify highly correlated pairs
    print("\n🔥 Highly Correlated Sensors (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                print(
                    f"  {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}"
                )

    return df


def detect_anomalies(df):
    """
    Detect potential anomalies in sensor readings.
    """
    print("\n" + "=" * 70)
    print("STEP 4: ANOMALY DETECTION")
    print("=" * 70)

    anomalies = []

    # Temperature anomalies (>85°C is concerning)
    temp_cols = ["Temp_Opposite_C", "Temp_Motor_C"]
    for col in temp_cols:
        high_temp = df[df[col] > 85]
        if len(high_temp) > 0:
            anomalies.append(
                {
                    "type": "High Temperature",
                    "sensor": col,
                    "count": len(high_temp),
                    "max_value": high_temp[col].max(),
                    "dates": high_temp["Date"].tolist()[:5],  # First 5
                }
            )

    # Vibration anomalies (>2.5 mm/s is concerning)
    vib_cols = ["Vibration_Opposite_mm_s", "Vibration_Motor_mm_s"]
    for col in vib_cols:
        high_vib = df[df[col] > 2.5]
        if len(high_vib) > 0:
            anomalies.append(
                {
                    "type": "High Vibration",
                    "sensor": col,
                    "count": len(high_vib),
                    "max_value": high_vib[col].max(),
                    "dates": high_vib["Date"].tolist()[:5],
                }
            )

    # Valve anomalies (sudden drops or extreme positions)
    valve_low = df[df["Valve_Opening_Percent"] < 10]
    if len(valve_low) > 0:
        anomalies.append(
            {
                "type": "Low Valve Opening",
                "sensor": "Valve_Opening_Percent",
                "count": len(valve_low),
                "min_value": valve_low["Valve_Opening_Percent"].min(),
                "dates": valve_low["Date"].tolist()[:5],
            }
        )

    print(f"\n🚨 Detected {len(anomalies)} anomaly patterns:\n")
    for i, anom in enumerate(anomalies, 1):
        print(f"{i}. {anom['type']} - {anom['sensor']}")
        print(f"   Occurrences: {anom['count']}")
        if "max_value" in anom:
            print(f"   Max value: {anom['max_value']:.2f}")
        if "min_value" in anom:
            print(f"   Min value: {anom['min_value']:.2f}")
        print(
            f"   Sample dates: {[d.strftime('%Y-%m-%d %H:%M') for d in anom['dates'][:3]]}"
        )
        print()

    return anomalies


def create_basic_features(df):
    """
    Create basic time-series features.
    """
    print("\n" + "=" * 70)
    print("STEP 5: FEATURE ENGINEERING (BASIC)")
    print("=" * 70)

    print("\n🔧 Creating features...")

    sensor_cols = [col for col in df.columns if col != "Date"]

    # Time-based features
    df["hour"] = df["Date"].dt.hour
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    print("  ✅ Time-based features (hour, day_of_week, month)")

    # Rolling statistics (1-hour window = 60 minutes)
    window_sizes = [10, 60, 240]  # 10min, 1hour, 4hours

    for window in window_sizes:
        print(f"  ⏱️  Computing {window}-minute rolling statistics...")
        for col in sensor_cols:
            # Rolling mean
            df[f"{col}_rolling_mean_{window}m"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
            # Rolling std
            df[f"{col}_rolling_std_{window}m"] = (
                df[col].rolling(window=window, min_periods=1).std()
            )
            # Rolling max
            df[f"{col}_rolling_max_{window}m"] = (
                df[col].rolling(window=window, min_periods=1).max()
            )

    # Rate of change (difference from previous reading)
    print("  📈 Computing rate of change...")
    for col in sensor_cols:
        df[f"{col}_diff_1m"] = df[col].diff(1)
        df[f"{col}_diff_10m"] = df[col].diff(10)

    # Temperature differential
    df["Temp_Differential"] = df["Temp_Motor_C"] - df["Temp_Opposite_C"]

    # Vibration ratio
    df["Vibration_Ratio"] = df["Vibration_Motor_mm_s"] / (
        df["Vibration_Opposite_mm_s"] + 0.001
    )  # Avoid division by zero

    print(f"\n✅ Feature engineering complete!")
    print(f"   Original features: {len(sensor_cols)}")
    print(f"   Total features now: {len([c for c in df.columns if c != 'Date'])}")

    return df


def save_processed_data(df, output_file="processed_sensor_data.csv"):
    """
    Save processed data to CSV.
    """
    print("\n" + "=" * 70)
    print("STEP 6: SAVING PROCESSED DATA")
    print("=" * 70)

    print(f"\n💾 Saving to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"✅ Saved {len(df):,} rows with {len(df.columns)} columns")

    file_size = pd.read_csv(output_file).memory_usage(deep=True).sum() / 1024**2
    print(f"📦 File size: {file_size:.2f} MB")


def load_failure_log(filepath=FAILURE_LOG):
    """
    Load failure log from Excel file.
    """
    print("\n" + "=" * 70)
    print("STEP 7: LOADING FAILURE LOG")
    print("=" * 70)

    try:
        print(f"\n📂 Loading failure log from: {filepath}")

        # Try to read Excel file
        failures = pd.read_excel(filepath)

        print(f"✅ Loaded {len(failures)} failure events")
        print(f"\nColumns: {failures.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(failures.head(10))

        return failures

    except Exception as e:
        print(f"❌ Error loading failure log: {e}")
        print("\n⚠️  You may need to:")
        print("   1. Install openpyxl: pip install openpyxl")
        print("   2. Manually inspect the Excel file")
        print("   3. Export it as CSV for easier processing")
        return None


def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 70)
    print("🏭 OCP PREDICTIVE MAINTENANCE - QUICK START ANALYSIS")
    print("=" * 70)
    print(f"\n⏰ Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Load sensor data
    df = load_sensor_data()

    # Step 2: Data quality check
    df = data_quality_check(df)

    # Step 3: Explore patterns
    df = explore_sensor_patterns(df)

    # Step 4: Detect anomalies
    anomalies = detect_anomalies(df)

    # Step 5: Feature engineering
    df = create_basic_features(df)

    # Step 6: Save processed data
    save_processed_data(df)

    # Step 7: Load failure log
    failures = load_failure_log()

    # Summary
    print("\n" + "=" * 70)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\n✅ Sensor data processed: {len(df):,} records")
    print(f"✅ Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"✅ Features created: {len(df.columns)} total columns")
    print(f"✅ Anomalies detected: {len(anomalies)} patterns")
    if failures is not None:
        print(f"✅ Failure events loaded: {len(failures)}")
    else:
        print(f"⚠️  Failure events: Need manual processing")

    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print("\n1. 📊 Open processed_sensor_data.csv in your analysis tool")
    print("2. 📑 Manually extract failure timestamps from Excel file")
    print("3. 🏷️  Create labels (pre-failure windows)")
    print("4. 🤖 Train baseline Random Forest model")
    print("5. 📈 Build dashboard for visualization")

    print(f"\n⏰ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✨ Ready for model development! Good luck with the hackathon! 🚀\n")

    return df, anomalies, failures


if __name__ == "__main__":
    # Run the analysis
    df, anomalies, failures = main()

    # Store in global namespace for interactive use
    print("\n💡 TIP: Run this in interactive mode (python -i quick_start_analysis.py)")
    print("   to access 'df', 'anomalies', and 'failures' variables for exploration.\n")
