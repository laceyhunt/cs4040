import final_data_expl
import tensorflow as tf
from tensorflow import keras
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = final_data_expl.make_df()
train_x, test_x, train_y, test_y, feature_names, outcome_names = final_data_expl.preprocess_data(df)

def train_model(model, epochs=100, batch_size=16, val_split=0.1, early_stop_patience=2):
   global train_x, train_y
   callback_fxn_1 = keras.callbacks.EarlyStopping(patience=early_stop_patience, monitor="val_loss", restore_best_weights=True)
   callback_fxn_2 = keras.callbacks.ModelCheckpoint(
      "my_model/checkpoint.models.keras",
      monitor="val_loss",
      mode="min", # because its loss, save minimum
      save_best_only=True
   )
   
   epochs = epochs
   batch_size = batch_size
   history = model.fit(x=train_x, 
                     y=train_y, 
                     batch_size=batch_size, 
                     epochs=epochs, 
                     callbacks=[callback_fxn_1, callback_fxn_2], 
                     validation_split=val_split)
   print(history.history)
   print(history.history.keys())

   train_return_dict = model.evaluate(x=train_x, y=train_y, verbose=0)
   print("Train: Loss, Accuracy")
   print(train_return_dict)
   print("Validation: Accuracy")
   
   # For multi class outcome
   # print(history.history["val_categorical_accuracy"])
   
   # For binary class outcome
   print(history.history["val_binary_accuracy"])

   test_return_dict = model.evaluate(x=test_x, y=test_y, verbose=0)
   print("Test: Loss, Accuracy")
   print(test_return_dict)
   
"""
   
# ---------------------------------------------
# Build the model
# ---------------------------------------------
model1 = keras.models.Sequential([keras.layers.Input(shape=(train_x.shape[-1], )), 
                                  keras.layers.Dense(30, activation = "relu"),
                                  keras.layers.Dense(30, activation = "relu"),
                                  keras.layers.Dense(1, activation="sigmoid")])                    # for binary class outcome
                                  #keras.layers.Dense(train_y.shape[-1], activation="softmax")])   # for multi class outcome
# Print model summary - helpful especially for shape debugging
print("  ##### MODEL 1: #####")
print(model1.summary())
# ---------------------------------------------
# Compile the model
# ---------------------------------------------
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
# For multi class outcome:
# loss_fxn = keras.losses.CategoricalCrossentropy()     
# metric_list = [keras.metrics.CategoricalAccuracy()]

# For binary class outcome
loss_fxn = keras.losses.BinaryCrossentropy()             
metric_list = [keras.metrics.BinaryAccuracy()]

model1.compile(optimizer=optimizer, loss=loss_fxn, metrics=metric_list)
train_model(model1, epochs=500,early_stop_patience=10)


"""

model = DecisionTreeClassifier()
finance_tree = model.fit(train_x, train_y)
test_accuracy_score = model.score(test_x, test_y)
training_accuracy_score = model.score(train_x, train_y)
print(f'Training Accuracy Score: {training_accuracy_score}')
print(f'Testing Accuracy Score: {test_accuracy_score}')


text_representation = tree.export_text(finance_tree)
# print(text_representation)
with open("decistion_tree.log", "w") as fout:
   fout.write(text_representation)
   
#code src: https://mljar.com/blog/visualize-decision-tree/

fig = final_data_expl.plt.figure(figsize=(25,20), dpi=300)
_ = tree.plot_tree(finance_tree,
                  #  feature_names=features,
                  #  class_names=outcomes,
                   filled=True)
fig.savefig("finance_decision_tree.png", dpi=300)