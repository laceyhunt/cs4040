import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.ensemble import RandomForestClassifier

faults = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

def read_in_dataset(faults):
   df = pd.read_csv('../faults.csv')
   df['fault'] = 0  # Sets all initially to 0
   # Create categorical variable for each fault
   for i in range(0,len(faults)):
      # Indexes of faults by type
      true_fault_idxs = df.loc[df[faults[i]]==1].index.tolist()
      df.loc[true_fault_idxs, 'fault'] = i+1
   return df

df = read_in_dataset(faults)
drop_features = ['fault'] + faults
features = df.drop(drop_features, axis=1)
outcomes = df['fault']
training_features, test_features, training_outcomes, test_outcomes = train_test_split(features, outcomes, test_size=.1)

# model = tree.DecisionTreeClassifier()# max_depth=3) # You can adjust a lot of these params
model = RandomForestClassifier()
model.fit(training_features, training_outcomes)
test_accuracy_score = model.score(test_features, test_outcomes)
training_accuracy_score = model.score(training_features, training_outcomes)
print(f'Training Accuracy Score: {training_accuracy_score}')
print(f'Testing Accuracy Score: {test_accuracy_score}')

# # Visualize the tree, wasnt working for me
# dot_data = tree.export_graphviz(model, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render('steel_tree')

# Predict a random variable
test_features.reset_index(inplace=True)
number = 3
random_features = pd.DataFrame([test_features.iloc[number]])
random_features = random_features.drop(['index'],axis=1)
actual_outcome = test_outcomes.tolist()[number]
outcome_prediction = model.predict(random_features)
# Print
print('Input to the prediction model:')
print(random_features)
print('Outcome:')
print(actual_outcome)
print('Prediction:')
print(outcome_prediction[0])
# Have to subtract one because we added one to our list of possible outcomes earlier (bc no fault is 0)
print(f'Predicted fault: {faults[outcome_prediction[0]-1]}, Actual fault: {faults[actual_outcome-1]}')

predicted_probabilities = model.predict_proba(random_features)
print(f'The predicted probabilities for each class were: {predicted_probabilities}')