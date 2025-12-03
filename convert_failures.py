#!/usr/bin/env python3
"""
Convert French failure log to failure_events.csv format
This processes the actual OCP failure data
"""

from datetime import datetime

import pandas as pd

print("=" * 70)
print("CONVERTING FAILURE LOG TO STANDARD FORMAT")
print("=" * 70)

# Load the French CSV
df = pd.read_csv("Suivi des arrêts_ligne 307_2019.csv")

print(f"\n✅ Loaded {len(df)} failure events")
print(f"\nColumns: {df.columns.tolist()}")

# Extract relevant columns
failures = []

for idx, row in df.iterrows():
    # Parse date and time
    date_str = row["Date de debut"]  # e.g., "1/7/2019"
    time_str = row["Heure de debut"]  # e.g., "21:00"

    # Convert to standard format
    try:
        # Handle dates like "1/7/2019" (M/D/YYYY)
        date_parts = date_str.split("/")
        month = int(date_parts[0])
        day = int(date_parts[1])
        year = int(date_parts[2])

        # Handle times like "21:00"
        time_parts = time_str.split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1])

        # Create datetime
        timestamp = datetime(year, month, day, hour, minute)

        # Get failure type from description
        description = row["Déscription"]
        failure_type = row["Type d'arret"] if "Type d'arret" in row else "MM"

        # Get duration
        duration = row["DUREE"]

        # Classify failure type based on description
        if "vibration" in description.lower():
            category = "Vibration"
        elif (
            "température" in description.lower() or "temperature" in description.lower()
        ):
            category = "Temperature"
        elif "palier" in description.lower():
            category = "Bearing"
        elif "huile" in description.lower():
            category = "Lubrication"
        elif "turbine" in description.lower():
            category = "Turbine"
        elif "programmé" in description.lower():
            category = "Scheduled"
        else:
            category = "Other"

        failures.append(
            {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
                "failure_type": category,
                "duration_hours": duration,
                "description": description.strip(),
            }
        )

        print(f"  ✓ {timestamp.strftime('%Y-%m-%d %H:%M')} - {category}")

    except Exception as e:
        print(f"  ⚠️  Skipped row {idx}: {e}")
        continue

# Create DataFrame
failures_df = pd.DataFrame(failures)

# Save to CSV
failures_df.to_csv("failure_events.csv", index=False)

print(f"\n✅ SUCCESS!")
print(f"   Created failure_events.csv with {len(failures)} events")
print(f"\n📊 Failure Types:")
print(failures_df["failure_type"].value_counts())

print("\n📅 Date Range:")
print(f"   First: {failures_df['timestamp'].iloc[0]}")
print(f"   Last: {failures_df['timestamp'].iloc[-1]}")

print("\n💾 Saved: failure_events.csv")
print("\n🚀 Now run: python3 ALL_IN_ONE.py")
