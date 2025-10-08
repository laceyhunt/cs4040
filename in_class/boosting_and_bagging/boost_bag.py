import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

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
training_features, test_features, training_outcomes, test_outcomes = train_test_split(features, outcomes, test_size=.1, random_state=4)

# model = tree.DecisionTreeClassifier()# max_depth=3) # You can adjust a lot of these params
# model = RandomForestClassifier(max_depth=10, n_estimators=5)
# model=BaggingClassifier(n_estimators=20)
# model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
model = GradientBoostingClassifier(n_estimators=50)
model.fit(training_features, training_outcomes)
test_accuracy_score = model.score(test_features, test_outcomes)
training_accuracy_score = model.score(training_features, training_outcomes)
print(f'Training Accuracy Score: {training_accuracy_score}')
print(f'Testing Accuracy Score: {test_accuracy_score}')
