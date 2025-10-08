import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt


df = pd.read_csv('../../datasets/fuel_end_use.csv')
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
   
   
# MODEL TYPE: DECISION TREE
print("\n\n  *** DECISION TREE MODEL ***")
print(' *** Original:')
model = tree.DecisionTreeClassifier()
gas_tree=model.fit(train_in, train_out)
text_representation = tree.export_text(gas_tree)
print(text_representation)
with open("decistion_tree.log", "w") as fout:
   fout.write(text_representation)
   
#code src: https://mljar.com/blog/visualize-decision-tree/

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(gas_tree,
                   feature_names=features,
                   class_names=outcomes,
                   filled=True)
fig.savefig("gas_decistion_tree.png")