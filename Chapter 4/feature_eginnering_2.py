# ===============================================================
# 0) IMPORTS & SETTINGS
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier

plt.style.use("fivethirtyeight")

# ===============================================================
# 1) LOAD PICKLE FILE
# ===============================================================

file = r"D:\PythonProjects\MachinelearningProjects\visualizations\03_final_cleaned_isolation_forest.pkl"
df = pd.read_pickle(file)

# Drop columns already used for magnitude features (they will be recomputed)
for col in ["acc_r", "gyr_r","set"]:
    if col in df.columns:
        del df[col]

# ===============================================================
# 2) TRAIN-TEST SPLIT
# ===============================================================

sensor_cols = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
categorical_cols = ["participant", "category"]  # your categorical features
target_col = "label"

# Numeric features
X_raw = df[sensor_cols].copy()
y = df[target_col].copy()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

# Categorical features (split same indices)
X_cat = df[categorical_cols].copy()
X_train_cat = X_cat.loc[X_train_raw.index]
X_test_cat  = X_cat.loc[X_test_raw.index]

# ===============================================================
# 3) LOW PASS FILTER
# ===============================================================

def low_pass_filter_causal(data, cols, fs=5.0, cutoff=1.2, order=4):
    df_filt = data.copy()
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    for col in cols:
        col_vals = df_filt[col].astype(float).values
        df_filt[col] = lfilter(b, a, col_vals)

    return df_filt

train_df = low_pass_filter_causal(X_train_raw, sensor_cols)
test_df  = low_pass_filter_causal(X_test_raw, sensor_cols)

# ===============================================================
# 4) TIME-DOMAIN FEATURES
# ===============================================================

def engineer_features(data, sensor_cols, window=5):
    df_feat = data.copy()

    # Magnitude features
    df_feat["acc_r"] = np.sqrt(df_feat["acc_x"]**2 + df_feat["acc_y"]**2 + df_feat["acc_z"]**2)
    df_feat["gyr_r"] = np.sqrt(df_feat["gyr_x"]**2 + df_feat["gyr_y"]**2 + df_feat["gyr_z"]**2)

    base_cols = list(sensor_cols) + ["acc_r", "gyr_r"]
    time_cols = base_cols.copy()

    for col in base_cols:
        roll = df_feat[col].rolling(window=window, min_periods=1)
        df_feat[col + "_mean"] = roll.mean()
        df_feat[col + "_std"]  = roll.std()
        df_feat[col + "_min"]  = roll.min()
        df_feat[col + "_max"]  = roll.max()
        time_cols += [col + "_mean", col + "_std", col + "_min", col + "_max"]

    df_feat = df_feat.ffill().fillna(0)
    return df_feat, time_cols

train_df, time_cols = engineer_features(train_df, sensor_cols)
test_df, _ = engineer_features(test_df, sensor_cols)

# ===============================================================
# 5) FFT FEATURES
# ===============================================================

def add_rfft_features(data, sensor_cols, fs=5.0, window_size=20, step=5):
    df = data.copy().reset_index(drop=True)
    n = len(df)
    half = window_size // 2

    print("   ⚡ RFFT Features...")

    for col in sensor_cols:
        signal = df[col].astype(float).values
        dom = np.full(n, np.nan)
        pw = np.full(n, np.nan)
        ent = np.full(n, np.nan)

        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            seg = signal[start:end]

            fft_vals = np.fft.rfft(seg)
            psd = (np.abs(fft_vals) ** 2) / window_size
            freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)
            psd_sum = psd.sum()

            if psd_sum <= 1e-12:
                dom_val, power_val, ent_val = 0.0, 0.0, 0.0
            else:
                dom_val = freqs[np.argmax(psd)]
                power_val = psd_sum
                pnorm = psd / psd_sum
                entropy = -np.sum(pnorm * np.log2(pnorm + 1e-12))
                ent_val = entropy / np.log2(len(pnorm))
                ent_val = float(ent_val)

            idx = start + half
            if idx >= n:
                idx = n - 1

            dom[idx] = dom_val
            pw[idx] = power_val
            ent[idx] = ent_val

        df[f"{col}_dom_freq"] = pd.Series(dom).ffill().bfill().fillna(0.0)
        df[f"{col}_power"] = pd.Series(pw).ffill().bfill().fillna(0.0)
        df[f"{col}_entropy"] = pd.Series(ent).ffill().bfill().fillna(0.0)

    fourier_cols = [f"{c}_{f}" for c in sensor_cols for f in ("dom_freq", "power", "entropy")]
    return df, fourier_cols

train_df, fourier_cols = add_rfft_features(train_df, sensor_cols)
test_df, _ = add_rfft_features(test_df, sensor_cols)

# ===============================================================
# 6) ONE-HOT ENCODE CATEGORICAL FEATURES
# ===============================================================

X_train_cat = pd.get_dummies(X_train_cat, prefix=categorical_cols).astype(int)
X_test_cat  = pd.get_dummies(X_test_cat, prefix=categorical_cols).astype(int)

# Align train/test one-hot columns
X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join="outer", axis=1, fill_value=0)

# ===============================================================
# 7) FINAL FEATURE LIST + CONCATENATE
# ===============================================================

X_train = pd.concat([train_df[time_cols + fourier_cols].reset_index(drop=True),
                     X_train_cat.reset_index(drop=True)], axis=1)

X_test  = pd.concat([test_df[time_cols + fourier_cols].reset_index(drop=True),
                     X_test_cat.reset_index(drop=True)], axis=1)

feature_cols = X_train.columns.tolist()
print("✅ Final feature count (with categorical):", len(feature_cols))
print("✅ Train shape:", X_train.shape)
print("✅ Test shape:", X_test.shape)

# ===============================================================
# 8) FEATURE IMPORTANCE (RANDOM FOREST)
# ===============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False).reset_index(drop=True)

# ===============================================================
# 9) VISUALIZATION
# ===============================================================

plt.figure(figsize=(10, max(5, 0.35 * len(feature_importance_df))))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"], color="lightcoral")
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Feature Importance (All Features)")
plt.tight_layout()
plt.show()

# ===============================================================
# 10) TOP FEATURES
# ===============================================================

print("\n🔝 Top 15 Most Important Features:")
print(feature_importance_df.head(15))

# ===============================================================
# 11) PREPARE FINAL DATASETS (WITH TARGET)
# ===============================================================

X_train_final = X_train.copy()
X_test_final  = X_test.copy()

X_train_final[target_col] = y_train.values
X_test_final[target_col]  = y_test.values

# ===============================================================
# 12) SAVE FILES
# ===============================================================

X_train_final.to_pickle("train_features_full.pkl")
X_test_final.to_pickle("test_features_full.pkl")
# Optional CSV
# X_train_final.to_csv("train_features_full.csv", index=False)
# X_test_final.to_csv("test_features_full.csv", index=False)
