import pandas as pd
import matplotlib.pyplot as plt
# Our inputs are continuous not discrete so we need gaussian... bc the outcomes (1 or 0, discrete) occur at a specific location which is continuous
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns

df = pd.read_csv('../faults.csv')
# print(df.head())

# Visualize the Correlation
# print(df.corr())
plot = sns.heatmap(df.corr())
# plt.savefig('defect_correlations.png')
# plt.clf()

faults = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
df['fault'] = 0  # Sets all initially to 0
# print(df.columns)

# We are going to number each of the faults and make a new column to notate which fault it has
# Create a categorical variable for each fault
# Inside the new 'fault' column
for i in range(0,len(faults)):
   true_fault_indexes = df.loc[df[faults[i]]==1].index.tolist()
   df.loc[true_fault_indexes, 'fault'] = i+1
   
# print(df['fault'])


# Create our dataset - inputs and outcomes
drop_features = ['fault']+faults
features = df.drop(drop_features,axis=1)
outcomes = df['fault']
# print(features.head())
# print(outcomes.head())

training_features, test_features, training_outcomes, test_outcomes = train_test_split(features, outcomes, test_size=.1)

bayes_classifier = GaussianNB()
bayes_classifier.fit(training_features, training_outcomes)
mean_accuracy = bayes_classifier.score(test_features, test_outcomes)
print(f'Mean accuracy of model: {mean_accuracy}')

# Predict a random row
test_features.reset_index(inplace=True)
# Input to prediction model
number = 3 # Randomly choose one
random_features = pd.DataFrame([test_features.iloc[number]])
random_features = random_features.drop(['index'], axis=1)
actual_outcome = test_outcomes.tolist()[number] # Actual defect of the input
outcome_prediction = bayes_classifier.predict(random_features)
# Print it out
print('Input to the prediction model:')
print(random_features)
print('Outcome:')
print(actual_outcome)
print('Prediction:')
print(outcome_prediction[0])
# Have to subtract one because we added one to our list of possible outcomes earlier (bc no fault is 0)
print(f'Predicted fault: {faults[outcome_prediction[0]-1]}, Actual fault: {faults[actual_outcome-1]}')

test_predictions = bayes_classifier.predict(test_features.drop(['index'],axis=1))
report = classification_report(test_outcomes, test_predictions)
# print(report)

# Confusion Matrix
matrix = confusion_matrix(test_outcomes, test_predictions)
print('Confusion Matrix')
print(matrix)