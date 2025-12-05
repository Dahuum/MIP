#!/usr/bin/env python3
"""
PYTORCH_DEEP_LEARNING.py - LSTM Neural Network using PyTorch
==============================================================
Pure Deep Learning - NO Random Forest, NO Traditional ML

Uses PyTorch (stable on macOS, uses all CPU cores)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score, f1_score

# Use all CPU cores
torch.set_num_threads(os.cpu_count())
device = torch.device('cpu')

print("=" * 80)
print("🧠 PyTorch LSTM DEEP LEARNING MODEL")
print("   Pure Neural Network - No Random Forest")
print("=" * 80)
print(f"   PyTorch Version: {torch.__version__}")
print(f"   CPU Threads: {torch.get_num_threads()}")
print(f"   Device: {device}")

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "SEQUENCE_LENGTH": 30,
    "PREDICTION_HOURS": 24,
    "HIDDEN_SIZE": 64,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.2,
    "EPOCHS": 30,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.001,
    "SKIP_ROWS": 15609,
    "SAMPLE_RATE": 5,
    "DUST_WARNING": 15.0,
}

print("\n⚙️  CONFIG:", CONFIG)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("📊 STEP 1: Loading data...")
print("=" * 80)

failures_df = pd.read_csv("failure_events_with_dust.csv")
failures_df["timestamp"] = pd.to_datetime(failures_df["timestamp"])
print(f"   ✓ {len(failures_df)} failures loaded")

df = pd.read_csv("Dataframemin.csv", skiprows=range(1, CONFIG["SKIP_ROWS"] + 1))
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M")
df.columns = ["Date", "Motor_Current", "Temp_Opposite", "Temp_Motor", 
              "Vib_Opposite", "Vib_Motor", "Valve_Opening"]
df = df.sort_values("Date").reset_index(drop=True)

# Sample for speed
df = df.iloc[::CONFIG["SAMPLE_RATE"]].reset_index(drop=True)
print(f"   ✓ {len(df):,} records (sampled)")

# Clean
sensor_cols = ["Motor_Current", "Temp_Opposite", "Temp_Motor", 
               "Vib_Opposite", "Vib_Motor", "Valve_Opening"]
for col in sensor_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[sensor_cols] = df[sensor_cols].ffill().bfill()

# =============================================================================
# STEP 2: FEATURES
# =============================================================================
print("\n" + "=" * 80)
print("🔧 STEP 2: Creating features...")
print("=" * 80)

feature_cols = sensor_cols.copy()

for col in sensor_cols:
    df[f"{col}_diff"] = df[col].diff().fillna(0)
    feature_cols.append(f"{col}_diff")

df["Vib_Total"] = df["Vib_Motor"] + df["Vib_Opposite"]
df["Temp_Diff"] = df["Temp_Motor"] - df["Temp_Opposite"]
feature_cols.extend(["Vib_Total", "Temp_Diff"])

print(f"   ✓ {len(feature_cols)} features")

# =============================================================================
# STEP 3: LABELS
# =============================================================================
print("\n" + "=" * 80)
print("🏷️  STEP 3: Creating labels...")
print("=" * 80)

df["label"] = 0
for _, f in failures_df.iterrows():
    start = f["timestamp"] - timedelta(hours=CONFIG["PREDICTION_HOURS"])
    mask = (df["Date"] >= start) & (df["Date"] < f["timestamp"])
    df.loc[mask, "label"] = 1

pos = (df["label"] == 1).sum()
neg = (df["label"] == 0).sum()
print(f"   ✓ Positive: {pos:,}, Negative: {neg:,}")

# =============================================================================
# STEP 4: PREPARE DATA
# =============================================================================
print("\n" + "=" * 80)
print("🔄 STEP 4: Preparing sequences...")
print("=" * 80)

scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

split = int(len(df) * 0.7)
train_df = df[:split]
test_df = df[split:]

def make_sequences(data, seq_len):
    X, y = [], []
    vals = data[feature_cols].values
    labs = data["label"].values
    for i in range(seq_len, len(data)):
        X.append(vals[i-seq_len:i])
        y.append(labs[i])
    return np.array(X), np.array(y)

seq_len = CONFIG["SEQUENCE_LENGTH"]
X_train, y_train = make_sequences(train_df, seq_len)
X_test, y_test = make_sequences(test_df, seq_len)

print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True)

# Class weights - cap at 10x to prevent gradient explosion
raw_weight = neg / pos if pos > 0 else 1.0
pos_weight = torch.tensor([min(raw_weight, 10.0)])  # Cap weight to prevent instability
print(f"   ✓ Positive class weight: {pos_weight.item():.2f} (raw: {raw_weight:.2f})")

# =============================================================================
# STEP 5: DEFINE LSTM MODEL
# =============================================================================
print("\n" + "=" * 80)
print("🧠 STEP 5: Building LSTM Neural Network...")
print("=" * 80)

class LSTMModel(nn.Module):
    """
    LSTM Deep Learning Model for Predictive Maintenance
    
    Architecture:
    - LSTM layers (learn temporal patterns)
    - Dropout (prevent overfitting)
    - Dense output (binary classification)
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        # NOTE: No sigmoid here - BCEWithLogitsLoss applies it internally
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        # Take last timestep output
        last_out = lstm_out[:, -1, :]
        # Dropout + Dense (raw logits)
        out = self.dropout(last_out)
        out = self.fc(out)
        return out.squeeze()  # Returns raw logits, not probabilities

model = LSTMModel(
    input_size=len(feature_cols),
    hidden_size=CONFIG["HIDDEN_SIZE"],
    num_layers=CONFIG["NUM_LAYERS"],
    dropout=CONFIG["DROPOUT"]
)

print(f"\n   📐 MODEL ARCHITECTURE:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n   ✓ Total parameters: {total_params:,}")

# =============================================================================
# STEP 6: TRAIN (Self-correction via Backpropagation)
# =============================================================================
print("\n" + "=" * 80)
print("🎓 STEP 6: Training LSTM (Self-correcting via Backpropagation)...")
print("=" * 80)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print("\n   Epoch  |  Loss   | Train Acc | Train Rec |  Progress")
print("   " + "-" * 55)

for epoch in range(CONFIG["EPOCHS"]):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    true_pos = 0
    actual_pos = 0
    
    for X_batch, y_batch in train_loader:
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)  # Raw logits
        
        # Calculate loss
        loss = criterion(outputs, y_batch)
        
        # BACKPROPAGATION - This is the self-correction!
        loss.backward()  # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
        probs = torch.sigmoid(outputs)  # Convert logits to probabilities
        preds = (probs > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        
        # Track recall
        true_pos += ((preds == 1) & (y_batch == 1)).sum().item()
        actual_pos += (y_batch == 1).sum().item()
    
    acc = correct / total
    avg_loss = total_loss / len(train_loader)
    train_recall = true_pos / actual_pos if actual_pos > 0 else 0
    
    # Update learning rate based on loss
    scheduler.step(avg_loss)
    
    bar = "█" * (epoch + 1) + "░" * (CONFIG["EPOCHS"] - epoch - 1)
    print(f"   {epoch+1:3d}    | {avg_loss:.4f} |  {acc:.2%}   |  {train_recall:.2%}   | {bar[:20]}")

print("\n   ✓ Training complete!")

# =============================================================================
# STEP 7: EVALUATE
# =============================================================================
print("\n" + "=" * 80)
print("📊 STEP 7: Evaluating model...")
print("=" * 80)

model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    y_pred_proba = torch.sigmoid(logits).numpy()  # Convert logits to probabilities
    y_pred = (y_pred_proba > 0.5).astype(int)

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred)

print(f"\n   🎯 METRICS:")
print(f"      Recall:    {recall:.2%}")
print(f"      Precision: {precision:.2%}")
print(f"      F1-Score:  {f1:.2%}")

# =============================================================================
# STEP 8: FAILURE-BASED EVALUATION
# =============================================================================
print("\n" + "=" * 80)
print("🎯 STEP 8: Checking each failure...")
print("=" * 80)

test_start = df.iloc[split]["Date"]
test_failures = failures_df[failures_df["timestamp"] >= test_start]

print(f"\n   Testing {len(test_failures)} failures:\n")

caught = 0
dust_caught = 0
total_dust = 0

for _, failure in test_failures.iterrows():
    f_time = failure["timestamp"]
    f_type = failure["failure_type"]
    dust = failure["Dust_Quantity_kg"]
    is_dust = dust >= CONFIG["DUST_WARNING"]
    
    if is_dust:
        total_dust += 1
    
    window_start = f_time - timedelta(hours=CONFIG["PREDICTION_HOURS"])
    test_dates = test_df.iloc[seq_len:]["Date"].values
    mask = (test_dates >= np.datetime64(window_start)) & (test_dates <= np.datetime64(f_time))
    
    if mask.sum() > 0:
        preds = y_pred[mask]
        probs = y_pred_proba[mask]
        
        detected = preds.sum() > 0
        max_prob = probs.max()
        
        if detected:
            caught += 1
            if is_dust:
                dust_caught += 1
            status = "✓ CAUGHT"
        else:
            status = "✗ MISSED"
        
        dust_icon = "🌫️" if is_dust else "  "
        print(f"   {dust_icon} {status} {f_time} ({f_type}) prob={max_prob:.1%}")

failure_recall = caught / len(test_failures) if len(test_failures) > 0 else 0
dust_recall = dust_caught / total_dust if total_dust > 0 else 0

print(f"\n   📊 RESULTS:")
print(f"      Failures: {caught}/{len(test_failures)} caught ({failure_recall:.0%})")
print(f"      Dust failures: {dust_caught}/{total_dust} ({dust_recall:.0%})")

# =============================================================================
# STEP 9: SAVE
# =============================================================================
print("\n" + "=" * 80)
print("💾 STEP 9: Saving model...")
print("=" * 80)

torch.save(model.state_dict(), "lstm_pytorch_model.pt")
print("   ✓ lstm_pytorch_model.pt")

import joblib
joblib.dump(scaler, "lstm_scaler.pkl")
joblib.dump(feature_cols, "lstm_features.pkl")
joblib.dump(CONFIG, "lstm_config.pkl")
print("   ✓ Saved scaler, features, config")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("🏆 PYTORCH LSTM COMPLETE!")
print("=" * 80)

print(f"""
✅ TRUE DEEP LEARNING (PyTorch):

   • LSTM learns TEMPORAL patterns from sequences
   • BACKPROPAGATION self-corrects weights each batch
   • GRADIENT DESCENT optimizes automatically
   • NO Random Forest, NO decision trees
   
   How it self-corrects each batch:
   ─────────────────────────────────
   1. Forward pass → prediction
   2. Loss calculation → error
   3. loss.backward() → compute gradients
   4. optimizer.step() → UPDATE WEIGHTS ← LEARNING!
   ─────────────────────────────────

📊 RESULTS:
   Failure Recall: {failure_recall:.0%} ({caught}/{len(test_failures)})
   Dust Detection: {dust_recall:.0%} ({dust_caught}/{total_dust})

💰 ROI: ${caught * 200000:,} saved
""")

print("⏰ Done:", datetime.now().strftime("%H:%M:%S"))
