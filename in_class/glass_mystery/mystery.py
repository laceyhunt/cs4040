import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


df = pd.read_csv('../datasets/glass_training_set.csv')
# print(df.head)
# print(df['END_USE'].unique())
outcomes = ['Type_of_glass']
features = ['RI', 'Na', 'Mg','Al','Si','K','Ca', 'Ba', 'Fe']


# # Convert to numeric
# for feature in features:
#    df[feature] = pd.to_numeric(df[feature], errors='coerce') # turns errors into nans

# Drop the bad values
df = df.dropna()

def prep_dataset(df, features, outcome):  # set random state here?
   feature_df = df[features]
   outcome_df = df[outcome]
   # Normalize the data
   # scaler = MinMaxScaler()
   # feature_df = scaler.fit_transform(feature_df)
   # Split into training and test splits
   train_in, train_out, test_in, test_out = train_test_split(feature_df, outcome_df, test_size=0.2, random_state=4)
   return train_in,test_in,train_out,test_out


train_in, train_out, test_in, test_out = prep_dataset(df,features,'Type_of_glass')
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
# model = RandomForestClassifier()
run_model(model, train_in, test_in, train_out, test_out)

test_df =  pd.read_csv('../datasets/glass_test_set.csv')
outcome_prediction = model.predict(test_df)

print('Input to the prediction model:')
print(test_df)
print("Predictions...")
print(outcome_prediction)

# did not normalize the train
print(f"Classes: {outcomes}")
predicted_probabilities = model.predict_proba(test_df)
print(f'The predicted probabilities for each class were:\n {predicted_probabilities}')