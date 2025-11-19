# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read in faults csv
df = pd.read_csv('../datasets/faults.csv')
faults = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
df['fault'] = 0  # Sets all initially to 0
# print(df.head())

# ---------------------------------------------
# Create model features and outcomes
# We don't have to make faults column this time
# ---------------------------------------------

# Features: drop everything but the faults
features = df.drop(faults, axis=1)
# Outcomes: just the faults
outcomes = df[faults]
# print(features.columns)
# print(features.shape)
# print(outcomes.columns)
# print(outcomes.shape)

# ---------------------------------------------
# Data Preprocessing
# ---------------------------------------------
# Normalize everything
scaler = MinMaxScaler()
# Fit both because model expects them to be in the same format (scaler converts to numpy array)
features = scaler.fit_transform(features)
outcomes = scaler.fit_transform(outcomes)
# Split into training and test sets
train_x, test_x, train_y, test_y = train_test_split(features, outcomes, test_size=0.2)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# ---------------------------------------------
# Build the model
# ---------------------------------------------
# First, input layer provide the shape of the data (not batches)
input_layer = keras.layers.Input(shape=(train_x.shape[-1], )) # recall train_x shape is (num samples, num features) = (1552, 28)... expecting 2d dimensions
# Layers 1 and 2 are dense layers
# Pass them: # nodes, activation function
layer_1 = keras.layers.Dense(30, activation = "relu")
layer_2 = keras.layers.Dense(15, activation = "relu")
# Output layer: has same number of nodes as outputs
output_layer = keras.layers.Dense(train_y.shape[-1], activation="softmax")
# Model is passed a list of layers
model = keras.models.Sequential([input_layer, layer_1, layer_2, output_layer])
# Print model summary - helpful especially for shape debugging
print(model.summary())

# ---------------------------------------------
# Compile the model
# ---------------------------------------------
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fxn = keras.losses.CategoricalCrossentropy()
metric_list = [keras.metrics.CategoricalAccuracy()]
model.compile(optimizer=optimizer, loss=loss_fxn, metrics=metric_list)


callback_fxn_1 = keras.callbacks.EarlyStopping(patience=2, monitor="val_loss", restore_best_weights=True)
callback_fxn_2 = keras.callbacks.ModelCheckpoint(
   "neural_net/my_model/checkpoint.models.keras",
   monitor="val_loss",
   mode="min", # because its loss
   save_best_only=True
)
# ---------------------------------------------
# Train the model - Biggest time sink!
# ---------------------------------------------
epochs = 5
batch_size = 16
history = model.fit(x=train_x, 
                    y=train_y, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    callbacks=[callback_fxn_1, callback_fxn_2], 
                    validation_split=0.1)
print(history.history)
print(history.history.keys())
print("Loss")
print(history.history["loss"])
print(history.history["val_loss"])
print("Validation Accuracy")
print(history.history["val_categorical_accuracy"])

train_return_dict = model.evaluate(x=train_x, y=train_y, verbose=0)
print("Train: Loss, Accuracy")
print(train_return_dict)

test_return_dict = model.evaluate(x=test_x, y=test_y, verbose=0)
print("Test: Loss, Accuracy")
print(test_return_dict)

predictions = model.predict(test_x)
# Convert back from one-hot vectors
pred_y = np.argmax(predictions,axis=1)
out_y = np.argmax(test_y, axis=1)

# Get the confusion matrix and classification
matrix = confusion_matrix(out_y, pred_y)
report = classification_report(out_y, pred_y)
class_accuracies = matrix.diagonal()/matrix.sum(axis=1)
print(faults)
print(class_accuracies)
print(report)