# Impoerting necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import norm
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize':16})
plt.rcParams['figure.dpi'] = 120
%config inlineBackend.figure_format = 'retina'
%config InlineBackend.figure_format = 'svg'

# load the file
file_path = r"D:\PythonProjects\MachinelearningProjects\visualizations\preprocessed_data_with_resultant.pkl"

# reading the file
df = pd.read_pickle(file_path)

# checking for info
df.info()

# sensor columns
sensor_columns = df.columns[:6].to_list()

groups = np.array_split(sensor_columns, 2)

for group in groups:
    df[list(group) + ['label']].boxplot(by='label', layout=(1, len(group)), figsize=(12,5))
    plt.suptitle("")
    plt.xlabel("Exercise Type")
    plt.ylabel("Sensor Values")
    plt.show()



n_cols = 3
n_rows = int(np.ceil(len(sensor_columns) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

labels = df['label'].unique()
palette = sns.color_palette("viridis", len(labels))
for i, col in enumerate(sensor_columns):
    ax = axes[i]

    # Plot overlapping histograms
    sns.histplot(
        data=df,
        x=col,
        hue='label',
        ax=ax,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.6,
        linewidth=1,
        palette="viridis",
        # Important: Ensure labels are generated cleanly
        kde=False
    )

    # Styling
    ax.set_title(f"Distribution: {col}", fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Sensor Value")
    ax.set_ylabel("Density")

    # --- FIX FOR THE WARNING ---
    # Explicitly get handles and labels from the axis
    handles, labels = ax.get_legend_handles_labels()

    if handles:
        # Create legend only if there are items to show
        ax.legend(handles, labels, title="Exercise", loc='upper right', fontsize=9, title_fontsize=10)
    else:
        # If no labels found (rare), remove potential empty legend space
        ax.get_legend().remove()


# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Sensor Value Distributions by Exercise Type", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

#--------------------------------------------------------------------------
# function to plot binary outliers
#----------------------------------------------------------------------------


def plot_binary_outliers(dataset, col, outlier_col, method_name="Unknown", label_name=None, reset_index=True):
    """
    Plots a column with outliers highlighted based on a binary outlier column.
    Args:
        dataset (pd.DataFrame): The input dataframe containing the data and outlier flags.
        col (str): The name of the column to plot.
        outlier_col (str): The name of the boolean column indicating outliers (True for outliers).
        method_name (str, optional): The name of the outlier detection method for the title. Defaults to "Unknown".
        label_name (str, optional): An additional label for the title. Defaults to None.
        reset_index (bool, optional): Whether to reset the index for plotting. Defaults to True.

        Returns:
        None: Displays a plot with normal points in blue and outliers in red.

    """

    # 1. Prepare Data
    plot_df = dataset.copy()
    plot_df[outlier_col] = plot_df[outlier_col].astype(bool)
    plot_df = plot_df.dropna(subset=[col])

    if plot_df.empty:
        print(f"No valid data to plot for '{method_name}' on column '{col}'.")
        return

    # 2. Reset Index
    if reset_index:
        plot_df = plot_df.reset_index(drop=True)

    # 3. Create Masks
    is_normal = ~plot_df[outlier_col]
    is_outlier = plot_df[outlier_col]

    # 4. Setup Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # --- THE KEY COLORS ---
    normal_color = "#4C72B0"  # The exact Seaborn Blue from the screenshot
    outlier_color = "#C9282D" # A crisp red for outliers

    # Plot Normal Data (Blue dots)
    ax.plot(
        plot_df.index[is_normal],
        plot_df.loc[is_normal, col],
        '.',
        label='no outlier ' + col,
        alpha=0.7,
        color=normal_color,
        markersize=4
    )

    # Plot Outliers (Red dots)
    # Note: The screenshot uses small red dots, not big X's, to match the density
    ax.plot(
        plot_df.index[is_outlier],
        plot_df.loc[is_outlier, col],
        '.',
        label='outlier ' + col,
        alpha=0.9,
        color=outlier_color,
        markersize=4
    )

    # 5. Styling to match screenshot exactly
    title_parts = []
    if label_name:
        title_parts.append(f"[{label_name}]")
    title_parts.append(f"{method_name}")
    title_parts.append(f": {col}")

    ax.set_title(" ".join(title_parts), fontsize=14, pad=10)
    ax.set_xlabel("samples", fontsize=12)
    ax.set_ylabel("value", fontsize=12)

    # Legend styling (Top center, two columns)
    ax.legend(
        loc='upper center',
        ncol=2,
        frameon=True,
        shadow=True,
        fancybox=True,
        fontsize=10
    )

    plt.tight_layout()
    plt.show()
#----------------------------------------------------------------------------------------
# Insert IQR Function
#---------------------------------------------------------------------------------------
def mark_outliers_iqr(dataset, col, output_col=None):
    """
    Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want to apply outlier detection to
        output_col (string, optional): Name of the output boolean column.
                                       If None, defaults to '{col}_outlier'.
    Finds Outliers using the Interquartile Range (IQR) method. A value is considered an outlier if it is below Q1 - 1.5*IQR or above Q3 + 1.5*IQR, where IQR = Q3 - Q1.


    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column.
    """
    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Determine output column name
    if output_col is None:
        output_col = f"{col}_outlier"

    # Apply logic
    dataset[output_col] = (dataset[col] < lower_bound) | (dataset[col] > upper_bound)

    return dataset

# specify the col
col = "acc_x"
df_iqr = mark_outliers_iqr(dataset=df , col=col,output_col=None)
plot_binary_outliers(dataset=df_iqr,col=col,outlier_col=col+"_outlier",reset_index=True,label_name=None,method_name="IQR Method")

# looping tthrough all the outlier columns
for col in sensor_columns:
    df_iqr = mark_outliers_iqr(dataset=df , col=col,output_col=None)
    plot_binary_outliers(dataset=df_iqr,col=col,outlier_col=col+"_outlier",reset_index=True,label_name=None,method_name="IQR Method")


#-------------------------------------------------------------------------------------
# Chauvenet's Criterion
#-------------------------------------------------------------------------------------

def mark_outliers_chauvenet(dataset, col, output_col=None):
    """
    Finds outliers using Chauvenet's Criterion.

    This method assumes the data follows a Normal Distribution. It flags a point
    as an outlier if the probability of its occurrence is less than 1/(2N).

    Args:
        dataset (pd.DataFrame): The input dataframe.
        col (str): The column name to check for outliers.
        output_col (str, optional): Name of the output boolean column.
                                    If None, defaults to '{col}_outlier'.
    Finds Outliers using Chauvenet's Criterion. A value is considered an outlier if the probability of its occurrence is less than 1/(2N), where N is the total number of observations.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column indicating outliers.
    """

    # 1. Create a copy to avoid modifying original data
    df = dataset.copy()

    # 2. Determine output column name
    if output_col is None:
        output_col = f"{col}_outlier"

    # 3. Calculate Statistics
    mean = df[col].mean()
    std = df[col].std()
    N = len(df)

    # Safety Check: If std is 0 (all values identical), no outliers exist
    if std == 0 or N == 0:
        df[output_col] = False
        return df

    # 4. Calculate Z-scores (absolute deviation from mean)
    z_scores = np.abs((df[col] - mean) / std)

    # 5. Calculate Probability (Two-tailed test)
    # norm.sf gives the survival function (1 - cdf), which is the tail probability.
    # Multiply by 2 because outliers can be on either side (low or high).
    probabilities = 2 * norm.sf(z_scores)

    # 6. Determine Chauvenet's Threshold
    # Criterion: A value is an outlier if P(value) < 1 / (2 * N)
    criterion = 1.0 / (2 * N)

    # 7. Apply Mask (Vectorized)
    df[output_col] = probabilities < criterion

    return df



for col in sensor_columns:
    df_chauvenet = mark_outliers_chauvenet(dataset=df , col=col,output_col=None)
    plot_binary_outliers(dataset=df_chauvenet,col=col,outlier_col=col+"_outlier",reset_index=True,label_name=None,method_name="Chauvenet's Criterion")




def mark_outliers_lof(dataset, feature_cols, contamination=0.01, n_neighbors=20, output_col=None):
    """
    Finds outliers using the Local Outlier Factor (LOF) algorithm.

    LOF detects anomalies based on local density deviation. It requires data scaling
    because it relies on distance calculations.

    Args:
        dataset (pd.DataFrame): The input dataframe.
        feature_cols (list): List of column names to use for detection (Multivariate).
        contamination (float): Expected proportion of outliers (e.g., 0.01 for 1%).
        n_neighbors (int): Number of neighbors to use for k-distance. Typical range: 20-50.
        output_col (str, optional): Name of the output boolean column.
                                    If None, defaults to 'lof_outlier'.
 finds 

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column indicating outliers.
    """

    # 1. Create a copy
    df = dataset.copy()

    # 2. Determine output column name
    if output_col is None:
        output_col = "lof_outlier"

    # Safety Check: Need enough data points for neighbors
    if len(df) <= n_neighbors:
        print(f"Warning: Dataset size ({len(df)}) <= n_neighbors ({n_neighbors}). LOF cannot run. Returning all False.")
        df[output_col] = False
        return df

    # 3. Extract and SCALE Data (CRITICAL STEP for LOF)
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Initialize and Fit LOF
    # novelty=False means we are detecting outliers in the training set itself
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        metric='minkowski',
        n_jobs=-1
    )

    # fit_predict returns -1 for outliers, 1 for inliers
    preds = lof.fit_predict(X_scaled)

    # 5. Create Boolean Mask
    df[output_col] = (preds == -1)

    return df



def mark_outliers_isolation_forest(dataset, feature_cols, contamination=0.01, n_estimators=100, random_state=42, output_col=None):
    """
    Finds outliers using the Isolation Forest algorithm.
    Isolation Forest works by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of that feature. This process is repeated recursively, creating a tree structure. Outliers are isolated more quickly than normal points, resulting in shorter path lengths in the tree.
    Args:
        dataset (pd.DataFrame): The input dataframe.
        feature_cols (list): List of column names to use for detection (Multivariate).
        contamination (float): Expected proportion of outliers (e.g., 0.01 for 1%).
        n_estimators (int): Number of trees in the forest. Higher = more stable but slower. Default 100.
        random_state (int): Ensures reproducibility.
        output_col (str, optional): Name of the output boolean column.
                                    If None, defaults to 'if_outlier'.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column indicating outliers.
    """

    # 1. Create a copy
    df = dataset.copy()

    # 2. Determine output column name
    if output_col is None:
        output_col = "if_outlier"

    # Safety Check: Need enough data points
    if len(df) < 10:
        print(f"Warning: Dataset size ({len(df)}) is too small for Isolation Forest. Returning all False.")
        df[output_col] = False
        return df

    # 3. Extract Data
    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Initialize and Fit Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    # fit_predict returns -1 for outliers, 1 for inliers
    preds = iso_forest.fit_predict(X_scaled)

    # 5. Create Boolean Mask
    df[output_col] = (preds == -1)

    return df



sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

df_if = mark_outliers_isolation_forest(dataset=df, feature_cols=sensor_columns, contamination=0.01, n_estimators=100, random_state=42, output_col=None)


for col in sensor_columns:
    plot_binary_outliers(dataset=df_if,col=col,outlier_col="if_outlier",reset_index=True,label_name=None,method_name="Isolation Forest")


df_lof = mark_outliers_lof(dataset=df, feature_cols=sensor_columns, contamination=0.01, n_neighbors=20, output_col=None)

for col in sensor_columns:
    plot_binary_outliers(dataset=df_lof,col=col,outlier_col="lof_outlier",reset_index=True,label_name=None,method_name="Local Outlier Factor (LOF)")

label = "squat"
for col in sensor_columns:
    df_iqr = mark_outliers_iqr(dataset=df[df['label'] == label], col=col, output_col=None)
    plot_binary_outliers(dataset=df_iqr, col=col, outlier_col=col+"_outlier", reset_index=True, label_name=label, method_name="IQR Method")

for col in sensor_columns:
    df_chauvenet = mark_outliers_chauvenet(dataset=df[df['label'] == label], col=col, output_col=None)
    plot_binary_outliers(dataset=df_chauvenet, col=col, outlier_col=col+"_outlier", reset_index=True, label_name=label, method_name="Chauvenet's Criterion")

df_if_label = mark_outliers_isolation_forest(dataset=df[df['label'] == label], feature_cols=sensor_columns, contamination=0.01, n_estimators=100, random_state=42, output_col=None)

for col in sensor_columns:
    plot_binary_outliers(dataset=df_if_label,col=col,outlier_col="if_outlier",reset_index=True,label_name=label,method_name="Isolation Forest")

df_lof_label = mark_outliers_lof(dataset=df[df['label'] == label], feature_cols=sensor_columns, contamination=0.01, n_neighbors=20, output_col=None)
for col in sensor_columns:
    plot_binary_outliers(dataset=df_lof_label,col=col,outlier_col="lof_outlier",reset_index=True,label_name=label,method_name="Local Outlier Factor (LOF)")



cleaned_df = df.copy()
sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
contamination_rate = 0.01

# Loop through each Exercise Label (Contextual Outlier Detection)
for label in cleaned_df['label'].unique():
    mask = cleaned_df['label'] == label
    group_data = cleaned_df.loc[mask]

    # Skip small groups
    if len(group_data) < 20:
        print(f" Skipping '{label}': Too few samples ({len(group_data)})")
        continue

    # Detect Outliers (Multivariate)
    result = mark_outliers_isolation_forest(
        dataset=group_data,
        feature_cols=sensor_columns,
        contamination=contamination_rate,
        output_col='if_outlier'
    )

    # Get indices of flagged rows
    outlier_indices = result.index[result['if_outlier']]

    if len(outlier_indices) > 0:
        print(f"  Found {len(outlier_indices)} outliers in '{label}'. Repairing...")

        # 2. Repair: Set to NaN then Interpolate
        for col in sensor_columns:
            cleaned_df.loc[outlier_indices, col] = np.nan
            cleaned_df[col] = cleaned_df[col].interpolate(method='linear')
            cleaned_df[col] = cleaned_df[col].bfill().ffill()
    else:
        print(f" No outliers found in '{label}'.")


print("\n Cleaning Complete!")
print(f"Original Shape: {df.shape}")
print(f"Cleaned Shape:  {cleaned_df.shape}")

# Save the final product
output_filename = '03_final_cleaned_isolation_forest.pkl'
cleaned_df.to_pickle(output_filename)
print(f" Final dataset saved to: {output_filename}")



def investigate_sample_outliers(df, sensor_cols, n_samples=2):
    """
    Runs Isolation Forest temporarily, picks random outliers, and plots them
    with their neighbors for manual inspection.
    """
    print("🔍 Running temporary detection for investigation...")

    # 1. Temporary Detection (Same logic as your main loop)
    df_temp = df.copy()
    df_temp['temp_outlier'] = False

    for label in df_temp['label'].unique():
        mask = df_temp['label'] == label
        group = df_temp.loc[mask]

        if len(group) < 20: continue

        # Run IF
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        preds = iso.fit_predict(group[sensor_cols])

        # Mark outliers in temp df
        outlier_indices = group.index[preds == -1]
        df_temp.loc[outlier_indices, 'temp_outlier'] = True

        # 2. Pick Random Samples to Inspect
        if len(outlier_indices) > 0:
            samples = np.random.choice(outlier_indices, size=min(n_samples, len(outlier_indices)), replace=False)

            for idx in samples:
                plot_outlier_context(df, idx, sensor_cols, label)

def plot_outlier_context(df, idx, sensor_cols, label_name, window=15):
    """Plots the outlier surrounded by neighbors."""
    try:
        loc = df.index.get_loc(idx)
    except KeyError: return

    start = max(0, loc - window)
    end = min(len(df), loc + window + 1)
    subset = df.iloc[start:end].reset_index(drop=True)
    rel_pos = loc - start

    fig, axes = plt.subplots(len(sensor_cols), 1, figsize=(14, 2.5 * len(sensor_cols)))
    if len(sensor_cols) == 1: axes = [axes]

    sns.set_theme(style="whitegrid")

    for i, col in enumerate(sensor_cols):
        ax = axes[i]
        # Plot neighbors
        ax.plot(subset.index, subset[col], '-o', color='steelblue', alpha=0.5, label='Normal Context')
        # Plot Suspect
        ax.plot(rel_pos, subset.loc[rel_pos, col], 'rx', markersize=15, markeredgewidth=3, label=f'SUSPECT (Idx: {idx})')

        ax.set_title(f"{label_name.upper()} | {col}")
        ax.legend(loc='upper right')

    plt.suptitle(f"Investigating Outlier at Index {idx} (Exercise: {label_name})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==========================================
# RUN THE INVESTIGATION
# ==========================================
sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

# This will generate plots for ~2 random outliers per exercise type
investigate_sample_outliers(df, sensor_columns, n_samples=2)


