from readline import redisplay
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("/Users/rishanthreddy/Desktop/Tracking barbell Exercises/data/interim/01_data_processed.pkl")



# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])


plt.plot(set_df["acc_y"].reset_index(drop=True))
# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig,ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True),label = label)
    plt.legend()
    plt.show()
    
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig,ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True),label = label)
    plt.legend()
    plt.show()
    




# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

fig,ax = plt.subplots()
category_df.head()

category_df.groupby(["category"],group_keys=True)["acc_y"].plot()

ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participants_df = df.query("label == 'bench'").sort_values("participant").reset_index()
fig,ax = plt.subplots()

participants_df.groupby(["participant"],group_keys=True)["acc_y"].plot()

ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend



