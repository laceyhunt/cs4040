import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
# Model libraries
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../datasets/fuel_end_use.csv')
# print(df.head)
# print(df['END_USE'].unique())
outcomes = ['Process Heating','CHP and/or Cogeneration Process','Conventional Boiler Use']
features = ['Coal', 'Diesel', 'Natural_gas','Other','Residual_fuel_oil','Temp_degC','Total']

# Get only a sample of the dataset
df = df.sample(n=30000)

# Convert to numeric
for feature in features:
   df[feature] = pd.to_numeric(df[feature], errors='coerce') # turns errors into nans

# Drop the bad values
df = df.dropna()

# Separate by end use
ph_df = df.loc[df['END_USE'] == outcomes[0]]
chp_df = df.loc[df['END_USE'] == outcomes[1]]
boil_df = df.loc[df['END_USE'] == outcomes[2]]

def prep_dataset(df, features, outcome):  # set random state here?
   feature_df = df[features]
   outcome_df = df[outcome]
   # Encode the labels
   encoder = LabelEncoder()
   outcome_df = encoder.fit_transform(outcome_df)
   print('Classes')
   print(encoder.classes_)
   # Normalize the data
   scaler = MinMaxScaler()
   feature_df = scaler.fit_transform(feature_df)
   # Split into training and test splits
   train_in, train_out, test_in, test_out = train_test_split(feature_df, outcome_df, test_size=0.2, random_state=4)
   return train_in,test_in,train_out,test_out


train_in, train_out, test_in, test_out = prep_dataset(df,features,'END_USE')
# print(train_in[0])

def run_model(model, train_in, test_in, train_out, test_out):
   model.fit(train_in,train_out)
   test_acc_score = model.score(test_in, test_out)
   train_acc_score = model.score(train_in,train_out)
   print(f'Training Accuracy: {train_acc_score}')
   print(f'Test Accuracy: {test_acc_score}')
   print('Per Class Accuracy')
   test_predictions = model.predict(test_in)
   matrix = confusion_matrix(test_out, test_predictions)
   class_accuracies = matrix.diagonal()/matrix.sum(axis=1)
   print(class_accuracies)
   
   
# MODEL TYPE: DECISION TREE
print("\n\n  *** DECISION TREE MODEL ***")
print(' *** Original:')
model = tree.DecisionTreeClassifier()
run_model(model, train_in, test_in, train_out, test_out)
print('\n *** Max Depth=3:')
model = tree.DecisionTreeClassifier(max_depth=3)
run_model(model, train_in, test_in, train_out, test_out)
print('\n *** Criterion=Entropy:')
model = tree.DecisionTreeClassifier(criterion='entropy')
run_model(model, train_in, test_in, train_out, test_out)

# MODEL TYPE: RANDOM FOREST
print("\n\n  *** RANDOM FOREST MODEL ***")
print(' *** Original:')
model = RandomForestClassifier()
run_model(model, train_in, test_in, train_out, test_out)
print('\n *** Max Depth=3:')
model = RandomForestClassifier(max_depth=3)
run_model(model, train_in, test_in, train_out, test_out)
print('\n *** N_Estimators=4:')
model = RandomForestClassifier(n_estimators=4)
run_model(model, train_in, test_in, train_out, test_out)

# MODEL TYPE: SVM
print("\n\n  *** SVM MODEL ***")
print(' *** RBF Kernel:')
model = SVC(kernel='rbf')
run_model(model, train_in, test_in, train_out, test_out)
print('\n *** Linear Kernel:')
model = SVC(kernel='linear')
run_model(model, train_in, test_in, train_out, test_out)
print('\n *** Sigmoid Kernel:')
model = SVC(kernel='sigmoid')
run_model(model, train_in, test_in, train_out, test_out)