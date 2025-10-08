import pandas as pd
import matplotlib.pyplot as plt
# Our inputs are continuous not discrete so we need gaussian... bc the outcomes (1 or 0, discrete) occur at a specific location which is continuous
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns


df = pd.read_csv('../in_class/faults.csv')
# print(df.head())

faults = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
df['fault'] = 0  # Sets all initially to 0


#    ALL THE POSSIBLE INPUT FEATURES
# X_Minimum,X_Maximum,Y_Minimum,Y_Maximum,Pixels_Areas,X_Perimeter,Y_Perimeter,
# Sum_of_Luminosity,Minimum_of_Luminosity,Maximum_of_Luminosity,Length_of_Conveyer,
# TypeOfSteel_A300,TypeOfSteel_A400,Steel_Plate_Thickness,Edges_Index,Empty_Index,
# Square_Index,Outside_X_Index,Edges_X_Index,Edges_Y_Index,Outside_Global_Index,
# LogOfAreas,Log_X_Index,Log_Y_Index,Orientation_Index,Luminosity_Index,SigmoidOfAreas

# We are going to number each of the faults and make a new column to notate which fault it has
# Create a categorical variable for each fault
# Inside the new 'fault' column
for i in range(0,len(faults)):
   true_fault_indexes = df.loc[df[faults[i]]==1].index.tolist()
   df.loc[true_fault_indexes, 'fault'] = i+1
   
# Create our dataset - inputs and outcomes
drop_features = ['fault']+faults
outcomes = df['fault']
# print(outcomes.head())


# ALL FEATURES (original in class example) USED AS INPUT
all_features = df.drop(drop_features,axis=1)
# print(all_features.head())
training_features_orig, test_features_orig, training_outcomes_orig, test_outcomes_orig = train_test_split(all_features, outcomes, test_size=.1)

bayes_original = GaussianNB()
bayes_original.fit(training_features_orig, training_outcomes_orig)
mean_accuracy = bayes_original.score(test_features_orig, test_outcomes_orig)
print(f'Mean accuracy of ORIGINAL model, 10% training: {mean_accuracy}')


# SUBSET OF INPUT FEATURES V1 (taking out all location information):
unused_features = ['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter','Y_Perimeter','Outside_X_Index','Edges_X_Index','Edges_Y_Index','Outside_Global_Index',]
subset_1 = all_features.drop(unused_features,axis=1)     # all_features had already dropped the outcomes
training_features_v1, test_features_v1, training_outcomes_v1, test_outcomes_v1 = train_test_split(subset_1, outcomes, test_size=.1)

bayes_v1 = GaussianNB()
bayes_v1.fit(training_features_v1, training_outcomes_v1)
mean_accuracy_v1 = bayes_v1.score(test_features_v1, test_outcomes_v1)
print(f'Mean accuracy of V1 model, 10% training: {mean_accuracy_v1}')


# SUBSET OF INPUT FEATURES V2 (just location information):
subset_2 = df[['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter','Y_Perimeter','Outside_X_Index','Edges_X_Index','Edges_Y_Index','Outside_Global_Index',]]
# print(v2_features.head())
training_features_v2, test_features_v2, training_outcomes_v2, test_outcomes_v2 = train_test_split(subset_2, outcomes, test_size=.1)
bayes_v2 = GaussianNB()
bayes_v2.fit(training_features_v2, training_outcomes_v2)
mean_accuracy_v2 = bayes_v2.score(test_features_v2, test_outcomes_v2)
print(f'Mean accuracy of V2 model: {mean_accuracy_v2}')


# VARYING THE TEST SIZE FOR V1
# 5% training
training_features_v1_5, test_features_v1_5, training_outcomes_v1_5, test_outcomes_v1_5 = train_test_split(subset_1, outcomes, test_size=.05)

bayes_v1_5 = GaussianNB()
bayes_v1_5.fit(training_features_v1_5, training_outcomes_v1_5)
mean_accuracy_v1_5 = bayes_v1_5.score(test_features_v1_5, test_outcomes_v1_5)
print(f'Mean accuracy of V1 model, 5% training: {mean_accuracy_v1_5}')

# 20% training
training_features_v1_20, test_features_v1_20, training_outcomes_v1_20, test_outcomes_v1_20 = train_test_split(subset_1, outcomes, test_size=.2)

bayes_v1_20 = GaussianNB()
bayes_v1_20.fit(training_features_v1_20, training_outcomes_v1_20)
mean_accuracy_v1_20 = bayes_v1_20.score(test_features_v1_20, test_outcomes_v1_20)
print(f'Mean accuracy of V1 model, 20% training: {mean_accuracy_v1_20}')


# GET ALL PREDICTIONS FOR V1, 20% TEST
test_features_v1_20.reset_index(inplace=True)  # reset the index values since they were random
test_predictions = bayes_v1_20.predict(test_features_v1_20.drop(['index'],axis=1))
report = classification_report(test_outcomes_v1_20, test_predictions)
print()
print()
print(faults)
print(report)

# Confusion Matrix
matrix = confusion_matrix(test_outcomes_v1_20, test_predictions)
print('Confusion Matrix:')
print(matrix)