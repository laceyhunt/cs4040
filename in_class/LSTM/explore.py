import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras 

df = pd.read_csv("../datasets/austria_power.csv")
# print(df.columns)
features = [
   "AT_load_actual_entsoe_transparency",
   "AT_load_forecast_entsoe_transparency",
   "AT_solar_generation_actual",
   "AT_wind_onshore_generation_actual"
]
outcome = "AT_price_day_ahead"

# Format timestamp, could skip but then plotting would get ugly
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])

# Explore the data: Nans? Weird patterns?
# print(df.isnull().sum())
# print(len(df.index))

# plt.plot(df["utc_timestamp"], df[features[1]])
# plt.plot(df["utc_timestamp"], df[outcome])
# plt.savefig("LSTM/graphs/expl.png")
# plt.clf()

# Filter to just the years we will use
df = df[(df["utc_timestamp"].dt.year >=2015 ) & (df["utc_timestamp"].dt.year <= 2017)]
# print(df.describe())
# print(df.isnull().sum())

df.fillna(method="bfill", inplace=True)
df.fillna(method="ffill", inplace=True)
# print(df.isnull().sum())

# Split our sets
train_df = df[(df["utc_timestamp"].dt.year >=2015 ) & (df["utc_timestamp"].dt.year <= 2016)]
test_df = df[(df["utc_timestamp"].dt.year >=2017 ) & (df["utc_timestamp"].dt.year <= 2017)]
# print(train_df.head())

# Create our features and outcomes from training and test
train_features = train_df[features]
train_outcome = train_df[outcome].values.reshape(-1,1)
test_features = test_df[features]
test_outcome = test_df[outcome].values.reshape(-1,1)
# print(train_features.describe())

# Optional - create an offset to predict farther in advance
# Will get lower accuracy but will actually be actionable
train_features = train_features.iloc[:-3]
train_outcome = train_outcome[3:]
test_features = test_features.iloc[:-3]
test_outcome = test_outcome[3:]

# Normalize data features
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.fit_transform(test_features)
train_outcome = scaler.fit_transform(train_outcome)
test_outcome = scaler.fit_transform(test_outcome)
# print(train_features)

generator = keras.preprocessing.sequence.TimeseriesGenerator(train_features, train_outcome, length=24, batch_size=32)
# One call of generator will give us 32 input samples to the NN,
# each sample will have 24 timestamps as input and one timestamp as output
# Ex: call 1 of generator:
# 32 samples
# Sample 1:
#  Timestamps 0-23, columns are features (input)
#  Timestamp 24, column is outcome (outcome)
# print(len(generator[0]))

lstm_layer = keras.layers.LSTM(100, activation="relu", input_shape=(24,len(features)))
dense_layer = keras.layers.Dense(1, activation="relu")
model = keras.models.Sequential([lstm_layer, dense_layer])
model.compile(optimizer="adam", loss="mse")
print(model.summary())
# model.fit_generator(generator, steps_per_epoch=1, epochs=50)
model.fit(generator, steps_per_epoch=1, epochs=50)

# Predict generator
predict_generator = keras.preprocessing.sequence.TimeseriesGenerator(test_features, test_outcome, length=24, batch_size=32)
predictions = model.predict(predict_generator)
score = model.evaluate(predict_generator)
actual_test_outcomes = test_outcome[24:]
print(f"Model has score {score}")

start_index = 0
end_index = 1000
times = test_df["utc_timestamp"].iloc[24:]
plt.plot(times[start_index:end_index], predictions[start_index:end_index],color="red", label="predictions")
plt.plot(times[start_index:end_index], actual_test_outcomes[start_index:end_index],color="blue", label="ground_truth")
plt.legend()
plt.savefig("LSTM/graphs/LSTM_scores_mse_loss.png")
plt.clf()
