import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("/Users/rishanthreddy/Desktop/Tracking barbell Exercises/data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
df = df[df ["label"] != "rest"]

acc_r= df["acc_x"] ** 2+ df["acc_y"]**2+ df ["acc_z"] **2 
gyr_r = df ["gyr_x"] **2+ df ["gyr_y"]**2+ df ["gyr_z"]** 2 
df["acc_r"]= np.sqrt(acc_r) 
df["gyr_r"] = np.sqrt(gyr_r)

#Split data

bench_df = df[df ["label"] == "bench"]

squat_df = df[df ["label"] == "squat"]

row_df = df[df ["label"] == "row"]

ohp_df = df[df ["label"] == "ohp"]

dead_df = df[df ["label"] == "dead"]

#Visualize data to identify patterns

plot_df = bench_df

plot_df[plot_df ["set"] == plot_df["set"].unique()[0]]["acc_x"].plot() 
plot_df[plot_df ["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()

plot_df[plot_df ["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()

plot_df[plot_df ["set"] == plot_df ["set"].unique()[0]]["acc_r"].plot()

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
LowPass =  LowPassFilter()



bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]

squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]

row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]

ohp_set =  ohp_df[ohp_df ["set"]== ohp_df ["set"].unique()[0]] 
dead_set = dead_df[dead_df ["set"] == dead_df ["set"].unique()[0]]

bench_set["acc_r"].plot()

column = "acc_r"

LowPass.low_pass_filter(

bench_set, col=column, sampling_frequency=5, cutoff_frequency=0.4, order=10 )[column + "_lowpass"].plot()


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):

    data = LowPass.low_pass_filter(

    dataset, col=column, sampling_frequency=5, cutoff_frequency=cutoff, order=10

)

    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)

    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()

    plt.plot(dataset [f"{column}_lowpass"])

    plt.plot(peaks [f" {column}_lowpass"], "o", color="red")

    ax.set_ylabel(f"{column}_lowpass")

    exercise = dataset ["label"].iloc[0].title()

    category = dataset ["category"].iloc[0].title()

    plt.title(f"{category} {exercise}: {len (peaks)} Reps")

    plt.show()

    return len(peaks)

count_reps(bench_set)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
