import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import  os
from IPython.display import display
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (20, 4)
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize':16})
plt.rcParams['figure.dpi'] = 300
%config inlineBackend.figure_format = 'retina'
%config InlineBackend.figure_format = 'svg'

# ladong the data
file_path = r"D:\PythonProjects\MachinelearningProjects\Preprocessed\preprocessed_data.pkl"

# reading the file
df = pd.read_pickle(file_path)
# check the first five items
df.head()

# plotting single variables
set_df = df[df['set']==1]

# plot
fig ,ax = plt.subplots()
plt.plot(set_df['acc_y'].reset_index(drop=True))
plt.title("Set 1 Samples")
plt.xlabel("Samples")
plt.ylabel("acc y")
plt.show()

# plotting for different execerisec
for label in df['label'].unique():
    subset = df[df['label']==label]
    # display the dataframe
    fig , ax = plt.subplots()
    plt.plot(subset['acc_y'].reset_index(drop=True),label = label)
    plt.title(f"{label.capitalize()} data")
    plt.xlabel("Samples")
    plt.ylabel("acc_y")
    plt.legend()
    plt.show()


# Compare heavy and medium sets for a single participaint
squat_E_df = df.query("label=='squat' and participant == 'D'").reset_index(drop=True)
fig ,ax = plt.subplots()
squat_E_df.groupby(["category"])['acc_y'].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

df['label'].unique()

# deadleft
deadlift_A_df = df.query("label=='dead' and participant=='A'").reset_index(drop=True)
fig ,ax = plt.subplots()
deadlift_A_df.groupby(['category'])['acc_y'].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

# Barbell row
row_B_df = df.query("label=='row' and participant =='E'").reset_index(drop=True)
fig ,ax = plt.subplots()
row_B_df.groupby(['category'])['acc_y'].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()


# comparing all the particpants
label ='bench'
bench_df = df.query(f"label=='{label}'").sort_values(by='participant').reset_index(drop=True)
fig ,ax = plt.subplots()
bench_df.groupby(['participant'])['acc_y'].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()


label = 'squat'
squat_df = df.query(f"label=='{label}'").sort_values('participant').reset_index(drop=True)
fig ,ax = plt.subplots()
squat_df.groupby(['participant'])['acc_y'].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

# plotting mutiplee axis
# plot mutiple axis
label = "squat"
participant = "A"
squat_A_df = df.query(f"label=='{label}' and participant=='{participant}'").reset_index()
fig, ax = plt.subplots()
squat_A_df[['acc_x','acc_y','acc_z']].plot(ax=ax)
ax.set_title("Squat Exercise - Participant A")
ax.set_ylabel("Acceleration")
ax.set_xlabel("Samples")
plt.legend()
plt.show()


label = 'squat'
squat_df = df.query(f"label=='{label}'").sort_values('participant').reset_index(drop=True)
fig ,ax = plt.subplots()
squat_df.groupby(['participant'])[['acc_x','acc_y','acc_z']].plot(ax=ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()


for label in df['label'].unique():
    for participant in df['participant'].unique():
        # create a dataframe
        all_axis_df = df.query(f"label=='{label}' and participant=='{participant}'").reset_index()
        fig, ax = plt.subplots()
        squat_A_df[['acc_x','acc_y','acc_z']].plot(ax=ax)
        ax.set_title(f" {label.capitalize()} Exercise - Participant {participant}")
        ax.set_ylabel("Acceleration")
        ax.set_xlabel("Samples")
        plt.legend()
        plt.show()

# create resultant acceleration
df["acc_r"] = np.sqrt(
    df["acc_x"]**2 +
    df["acc_y"]**2 +
    df["acc_z"]**2
)

for label in df['label'].unique():
    for participant in df['participant'].unique():

        all_axis_df = df.query(
            f"label=='{label}' and participant=='{participant}'"
        ).reset_index()

        if all_axis_df.empty:
            continue

        fig, ax = plt.subplots()

        all_axis_df["acc_r"].plot(ax=ax)

        ax.set_title(f"{label.capitalize()} Exercise - Participant {participant}")
        ax.set_ylabel("Acceleration Magnitude")
        ax.set_xlabel("Samples")

        plt.show()


for label in df['label'].unique():
    for participant in df['participant'].unique():

        all_axis_df = df.query(
            f"label=='{label}' and participant=='{participant}'"
        ).reset_index()

        if all_axis_df.empty:
            continue

        fig, ax = plt.subplots()

        all_axis_df[['gyr_x','gyr_y','gyr_z']].plot(ax=ax)

        ax.set_title(f"{label.capitalize()} Exercise - Participant {participant}")
        ax.set_ylabel("Angular Velocity")
        ax.set_xlabel("Samples")

        plt.show()


df["gyr_r"] = np.sqrt(
    df["gyr_x"]**2 +
    df["gyr_y"]**2 +
    df["gyr_z"]**2
)

for label in df['label'].unique():
    for participant in df['participant'].unique():

        all_axis_df = df.query(
            f"label=='{label}' and participant=='{participant}'"
        ).reset_index()

        if all_axis_df.empty:
            continue

        fig, ax = plt.subplots()

        all_axis_df["gyr_r"].plot(ax=ax)

        ax.set_title(f"{label.capitalize()} Exercise - Participant {participant}")
        ax.set_ylabel("Angular Velocity")
        ax.set_xlabel("Samples")

        plt.show()


for label in df['label'].unique():
    for participant in df['participant'].unique():

        all_axis_df = df.query(
            f"label=='{label}' and participant=='{participant}'"
        ).reset_index()

        if all_axis_df.empty:
            continue

        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()

        all_axis_df["acc_r"].plot(
            ax=ax1,
            color="blue",
            label="Acceleration Magnitude"
        )

        all_axis_df["gyr_r"].plot(
            ax=ax2,
            color="red",
            label="Gyroscope Magnitude"
        )

        ax1.set_xlabel("Samples")
        ax1.set_ylabel("Acceleration", color="blue")
        ax2.set_ylabel("Angular Velocity", color="red")

        ax1.set_title(f"{label.capitalize()} Exercise - Participant {participant}")

        plt.show()

categories = ["heavy", "medium"]

for label in df["label"].unique():
    for category in categories:
        for participant in df["participant"].unique():

            subset = df.query(
                f"label=='{label}' and category=='{category}' and participant=='{participant}'"
            ).reset_index()

            if subset.empty:
                continue

            fig, ax1 = plt.subplots()

            ax2 = ax1.twinx()

            subset["acc_r"].plot(
                ax=ax1,
                color="blue",
                label="Acceleration Magnitude"
            )

            subset["gyr_r"].plot(
                ax=ax2,
                color="red",
                label="Gyroscope Magnitude"
            )

            ax1.set_xlabel("Samples")
            ax1.set_ylabel("Acceleration", color="blue")
            ax2.set_ylabel("Angular Velocity", color="red")

            title = f"{label.capitalize()} - {category.capitalize()} - Participant {participant}"
            ax1.set_title(title)

            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            plt.show()



# save the file

df.to_pickle(r"D:\PythonProjects\MachinelearningProjects\visualizations\preprocessed_data_with_resultant.pkl")
