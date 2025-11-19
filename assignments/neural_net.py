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

# ---------------------------------------------
# Build the model
# ---------------------------------------------
model1 = keras.models.Sequential([keras.layers.Input(shape=(train_x.shape[-1], )), 
                                  keras.layers.Dense(30, activation = "relu"),
                                  keras.layers.Dense(30, activation = "relu"),
                                  keras.layers.Dense(train_y.shape[-1], activation="softmax")])

model2 = keras.models.Sequential([keras.layers.Input(shape=(train_x.shape[-1], )),
                                  keras.layers.Dense(15, activation = "relu"),
                                  keras.layers.Dense(15, activation = "relu"), 
                                  keras.layers.Dense(train_y.shape[-1], activation="softmax")])

model3 = keras.models.Sequential([keras.layers.Input(shape=(train_x.shape[-1], )), 
                                  keras.layers.Dense(30, activation = "relu"),
                                  keras.layers.Dense(15, activation = "relu"),
                                  keras.layers.Dense(train_y.shape[-1], activation="softmax")])

# Print model summary - helpful especially for shape debugging
print("  ##### MODEL 1: #####")
print(model1.summary())
print("  ##### MODEL 2: #####")
print(model2.summary())
print("  ##### MODEL 3: #####")
print(model3.summary())

# ---------------------------------------------
# Compile the model
# ---------------------------------------------
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fxn = keras.losses.CategoricalCrossentropy()
metric_list = [keras.metrics.CategoricalAccuracy()]
model1.compile(optimizer=optimizer, loss=loss_fxn, metrics=metric_list)
model2.compile(optimizer=optimizer, loss=loss_fxn, metrics=metric_list)
model3.compile(optimizer=optimizer, loss=loss_fxn, metrics=metric_list)

# ---------------------------------------------
# Train the model - Biggest time sink!
# ---------------------------------------------


"""Try three different neural network architectures

   Different number of layers, or
   Different number of nodes, or
   Different activation function
"""

def train_model(model):
   global train_x, train_y
   callback_fxn_1 = keras.callbacks.EarlyStopping(patience=2, monitor="val_loss", restore_best_weights=True)
   callback_fxn_2 = keras.callbacks.ModelCheckpoint(
      "my_model/checkpoint.models.keras",
      monitor="val_loss",
      mode="min", # because its loss
      save_best_only=True
   )
   
   epochs = 100
   batch_size = 16
   history = model.fit(x=train_x, 
                     y=train_y, 
                     batch_size=batch_size, 
                     epochs=epochs, 
                     callbacks=[callback_fxn_1, callback_fxn_2], 
                     validation_split=0.1)
   print(history.history)
   print(history.history.keys())

   train_return_dict = model.evaluate(x=train_x, y=train_y, verbose=0)
   print("Train: Loss, Accuracy")
   print(train_return_dict)
   print("Validation: Accuracy")
   print(history.history["val_categorical_accuracy"])
   test_return_dict = model.evaluate(x=test_x, y=test_y, verbose=0)
   print("Test: Loss, Accuracy")
   print(test_return_dict)

# train_model(model1)
# train_model(model2)
train_model(model3)
