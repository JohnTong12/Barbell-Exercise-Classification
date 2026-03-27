# ===============================================================
# 0) IMPORTS & SETTINGS
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             f1_score, matthews_corrcoef)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

plt.style.use("fivethirtyeight")

# ===============================================================
# 1) LOAD DATA
# ==============================================================

train_file_path = r"D:\PythonProjects\MachinelearningProjects\Feature_Eginnering\train_features_full.pkl"
test_file_path  = r"D:\PythonProjects\MachinelearningProjects\Feature_Eginnering\test_features_full.pkl"

X_train = pd.read_pickle(train_file_path)
X_test  = pd.read_pickle(test_file_path)

print(f"X_train Shape : {X_train.shape}")
print(f"X_test  Shape : {X_test.shape}")
print(f"Classes       : {X_train['label'].unique()}\n")

# ===============================================================
# 2) CLASSICAL ML PREP
# ==============================================================

X_train_classical = X_train.drop(columns=['label'])
X_test_classical  = X_test.drop(columns=['label'])

# Label encoding
le = LabelEncoder()
y_train_classical = le.fit_transform(X_train['label'])
y_test_classical  = le.transform(X_test['label'])

print(f"Classes mapping : {dict(enumerate(le.classes_))}")

# ==============================
# ✅ FIXED SCALING (SIMPLE)
# ==============================

scaler = StandardScaler()

# Select numeric columns only
num_cols = [col for col in X_train_classical.columns
            if X_train_classical[col].nunique() > 2]

# Copy data
X_train_scaled = X_train_classical.copy()
X_test_scaled  = X_test_classical.copy()

# Scale only numeric columns
X_train_scaled[num_cols] = scaler.fit_transform(X_train_scaled[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test_scaled[num_cols])

# Convert to numpy for models
X_train_classical = X_train_scaled.values
X_test_classical  = X_test_scaled.values

# ===============================================================
# 3) QUICK VISUAL CHECK
# ==============================================================

df_plot = X_train.sample(n=5000, random_state=42)

sns.pairplot(
    df_plot,
    hue='label',
    vars=[
        "acc_x_mean",
        "acc_y_std",
        "gyr_x_entropy",
        "acc_r"
    ]
)

print("Train label distribution:\n", X_train['label'].value_counts(normalize=True))
print("Test label distribution:\n", X_test['label'].value_counts(normalize=True))

# ===============================================================
# 4) DEEP LEARNING PREP
# ==============================================================

X_train_dl = X_train_classical.copy()
X_test_dl  = X_test_classical.copy()

y_train_dl = to_categorical(y_train_classical)
y_test_dl  = to_categorical(y_test_classical)

# ===============================================================
# 5) MODEL RUNNER
# ==============================================================

metrics_log = []

def run_model(model, model_name, X_train, y_train, X_test, y_test, decimals=2):

    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    results = {
        "Model"      : model_name,
        "Accuracy"   : accuracy_score(y_test, y_hat) * 100,
        "Error Rate" : (1 - accuracy_score(y_test, y_hat)) * 100,
        "Precision"  : precision_score(y_test, y_hat, average='weighted') * 100,
        "Recall"     : recall_score(y_test, y_hat, average='weighted') * 100,
        "F1 Score"   : f1_score(y_test, y_hat, average='weighted') * 100,
        "MCC"        : matthews_corrcoef(y_test, y_hat),
    }

    results = {k: (round(v, decimals) if isinstance(v, float) else v)
               for k, v in results.items()}

    cm = pd.DataFrame(
        confusion_matrix(y_test, y_hat),
        index  =[f"Actual: {c}"  for c in le.classes_],
        columns=[f"Predict: {c}" for c in le.classes_]
    )

    print(f"\n{'═'*55}")
    print(f"  {model_name}")
    print(f"{'═'*55}")
    print(cm.to_string())
    print(f"{'─'*55}")
    for metric, value in results.items():
        if metric != "Model":
            print(f"  {metric:<14} = {value}%")
    print(f"{'═'*55}")

    metrics_log.append(results)

# ===============================================================
# 6) RUN MODELS
# ==============================================================

run_model(LogisticRegression(max_iter=1000, random_state=42),
          "Logistic Regression",
          X_train_classical, y_train_classical,
          X_test_classical,  y_test_classical)

run_model(SVC(C=1, gamma='scale', kernel='rbf'),
          "SVM (RBF)",
          X_train_classical, y_train_classical,
          X_test_classical,  y_test_classical)

run_model(KNeighborsClassifier(n_neighbors=5),
          "KNN",
          X_train_classical, y_train_classical,
          X_test_classical,  y_test_classical)

run_model(RandomForestClassifier(n_estimators=200, random_state=42),
          "Random Forest",
          X_train_classical, y_train_classical,
          X_test_classical,  y_test_classical)

run_model(XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss'),
          "XGBoost",
          X_train_classical, y_train_classical,
          X_test_classical,  y_test_classical)

run_model(LGBMClassifier(n_estimators=200, random_state=42),
          "LightGBM",
          X_train_classical, y_train_classical,
          X_test_classical,  y_test_classical)

# ===============================================================
# 7) FINAL COMPARISON
# ==============================================================

metrics_df = pd.DataFrame(metrics_log).set_index("Model")

print("\n📊 FINAL MODEL COMPARISON:\n")
print(metrics_df.sort_values(by="F1 Score", ascending=False))


# ===============================================================
# 6.5) DEEP LEARNING MODEL
# ==============================================================


def build_dnn(input_dim, num_classes):

    inputs = keras.Input(shape=(input_dim,))

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.ReLU()(x)

    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.ReLU()(x)

    x = layers.Dense(16, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.ReLU()(x)

    outputs = layers.Dense(len(le.classes_), activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ==============================
# PREP DATA
# ==============================

X_train_dl = X_train_classical.copy()
X_test_dl  = X_test_classical.copy()

y_train_dl = to_categorical(y_train_classical)
y_test_dl  = to_categorical(y_test_classical)

# # Convert to float (VERY IMPORTANT for DL)
# X_train_classical = X_train_scaled.astype(np.float32).values
# X_test_classical  = X_test_scaled.astype(np.float32).values


# y_train_dl = y_train_dl.astype(np.float32)
# y_test_dl  = y_test_dl.astype(np.float32)
# ==============================
# TRAIN
# ==============================

model = build_dnn(input_dim=X_train_dl.shape[1],
                  num_classes=len(le.classes_))

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_dl, y_train_dl,
    validation_data=(X_test_dl, y_test_dl),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# EVALUATE
# ==============================

y_pred_probs = model.predict(X_test_dl)
y_pred = np.argmax(y_pred_probs, axis=1)

results = {
    "Model"      : "Deep Learning (DNN)",
    "Accuracy"   : accuracy_score(y_test_classical, y_pred) * 100,
    "Error Rate" : (1 - accuracy_score(y_test_classical, y_pred)) * 100,
    "Precision"  : precision_score(y_test_classical, y_pred, average='weighted') * 100,
    "Recall"     : recall_score(y_test_classical, y_pred, average='weighted') * 100,
    "F1 Score"   : f1_score(y_test_classical, y_pred, average='weighted') * 100,
    "MCC"        : matthews_corrcoef(y_test_classical, y_pred),
}

results = {k: (round(v, 2) if isinstance(v, float) else v)
           for k, v in results.items()}

cm = pd.DataFrame(
    confusion_matrix(y_test_classical, y_pred),
    index  =[f"Actual: {c}"  for c in le.classes_],
    columns=[f"Predict: {c}" for c in le.classes_]
)

print(f"\n{'═'*55}")
print(f"  Deep Learning (DNN)")
print(f"{'═'*55}")
print(cm.to_string())
print(f"{'─'*55}")
for metric, value in results.items():
    if metric != "Model":
        print(f"  {metric:<14} = {value}%")
print(f"{'═'*55}")

metrics_log.append(results)
pd.DataFrame()

metrics_df = pd.DataFrame(metrics_log).set_index("Model")

print("\n📊 FINAL MODEL COMPARISON:\n")
print(metrics_df.sort_values(by="F1 Score", ascending=False))
