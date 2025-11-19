import pandas as pd
import random
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Read in data
sensor_1_df = pd.read_csv("../datasets/sensor_1.csv")
sensor_1_df["timestamp"] = pd.to_datetime(sensor_1_df["timestamp"])
sensor_1_df["timestamp"] = sensor_1_df["timestamp"].dt.tz_localize('utc')
sensor_1_df.drop(columns=["Unnamed: 0"], inplace=True)
# print(sensor_1_df.head())

# sensors 2 and 3 don't have timestamps
sensor_2_df = pd.read_csv("../datasets/sensor_2.csv")
sensor_2_df.drop(columns=["Unnamed: 0"], inplace=True)
# print(sensor_2_df.head())

sensor_3_df = pd.read_csv("../datasets/sensor_3.csv")
sensor_3_df.drop(columns=["Unnamed: 0"], inplace=True)
# print(sensor_3_df.head())

sensor_4_df = pd.read_csv("../datasets/sensor_4.csv")
sensor_4_df["timestamp"] = pd.to_datetime(sensor_4_df["timestamp"])
sensor_4_df["timestamp"] = sensor_4_df["timestamp"].dt.tz_localize('utc')
sensor_4_df.drop(columns=["Unnamed: 0"], inplace=True)
# print(sensor_4_df.head())

# Step 1: associating time stamp with the data that currently has no time stamp
# Generally, we want the time stamps in the same format

# Sensor 2: define a timestamp and range
start_sensor_2 = pd.Timestamp(2022, 1, 1, 1, 0, 0)
idx = pd.date_range(start_sensor_2, periods=len(sensor_2_df.index), freq="5min")
s2_readings = sensor_2_df.iloc[:,0].values.tolist()
df_dict = {"timestamp": idx, "reading": s2_readings}
s2_df = pd.DataFrame(df_dict)
s2_df["timestamp"] = s2_df["timestamp"].dt.tz_localize("utc")
# print(s2_df.head())

# Sensor 3: associate a time stamp here too
end_sensor_3 = pd.Timestamp(2022, 12, 31, 23, 8, 0)
idx = pd.date_range(end=end_sensor_3, periods=len(sensor_3_df.index), freq="5min")
s3_readings = sensor_3_df.iloc[:,0].values.tolist()
df_dict = {"timestamp": idx, "reading": s3_readings}
s3_df = pd.DataFrame(df_dict)
# Align sensor 3 timezone
s3_df["timestamp"] = s3_df["timestamp"].dt.tz_localize("prc")
s3_df["timestamp"] = s3_df["timestamp"].dt.tz_convert("utc")
# print(s3_df.head())

# # Check for missing data
# plt.scatter(sensor_1_df["timestamp"],sensor_1_df["reading"], color="g")
# plt.scatter(s2_df["timestamp"],s2_df["reading"], color="b")
# plt.scatter(s3_df["timestamp"],s3_df["reading"], color="r")
# plt.scatter(sensor_4_df["timestamp"],sensor_4_df["reading"], color="y")

# plt.savefig("time_series/readings.png")
# plt.clf()

# Note: the yellow (sensor 4) is missing like 3 months of data so we have to deal with that.
# So let's use Jan-March
start_date = pd.Timestamp(2022,1,1,0,0,0).tz_localize("utc")
start_date = pd.Timestamp(2022,3,31,23,0,0).tz_localize("utc")
# Decide our sampling interval: hourly
# min, max, mean for higher resolution sensors

# Start with sensor 2...
s2_min = s2_df.resample("h", on="timestamp", origin=start_date).min()
s2_min.reset_index(inplace=True)
s2_max = s2_df.resample("h", on="timestamp", origin=start_date).max()
s2_max.reset_index(inplace=True)
s2_mean = s2_df.resample("h", on="timestamp", origin=start_date).mean()
s2_mean.reset_index(inplace=True)

# Now 3
s3_min = s3_df.resample("h", on="timestamp", origin=start_date).min()
s3_min.reset_index(inplace=True)
s3_max = s3_df.resample("h", on="timestamp", origin=start_date).max()
s3_max.reset_index(inplace=True)
s3_mean = s3_df.resample("h", on="timestamp", origin=start_date).mean()
s3_mean.reset_index(inplace=True)
print(s3_mean.head())
print(s3_min.head())
print(s3_max.head())

# and 4
s4_df = sensor_4_df
s4_min = s4_df.resample("h", on="timestamp", origin=start_date).min()
s4_min.reset_index(inplace=True)
s4_max = s4_df.resample("h", on="timestamp", origin=start_date).max()
s4_max.reset_index(inplace=True)
s4_mean = s4_df.resample("h", on="timestamp", origin=start_date).mean()
s4_mean.reset_index(inplace=True)

# Create our merged df
new_df = s4_mean.copy()
new_df = new_df.rename(columns={"reading": "s4_mean"})
# print(new_df.head())

merge_options = [s2_min, s2_max, s2_mean, s3_min, s3_max, s3_mean,s4_min,s4_max] # S4 mean is already there
merge_names = ["s2_min", "s2_max", "s2_mean", "s3_min", "s3_max", "s3_mean","s4_min","s4_max"] # S4 mean is already there
for i in range(0, len(merge_options)):
   sub_df = merge_options[i]
   # This backfills
   sub_df.fillna(method="bfill", inplace=True)
   # This forward fills if you have no end one to backfill from
   sub_df.fillna(method="ffill", inplace=True)
   # so it should not need to drop any here but we run this just in case
   sub_df.dropna(inplace=True)
   new_df = pd.merge_asof(new_df, sub_df, direction="nearest")
   new_df = new_df.rename(columns={"reading": f"{merge_names[i]}"})

print(new_df.head())