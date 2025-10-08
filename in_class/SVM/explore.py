# Suppprt Vector Machine Ex with a dates dataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('../dates.csv')
# print(df.columns)
# plt.scatter(df['date_length'], df['date_diameter'])
# plt.savefig('init_scatter.png')
# plt.clf()

date_classes = ['Ajwa', "Medjool"]
df["class_code"] = 0
medjool_indexes = df.loc[df["class"] == "Medjool"].index.tolist()
df.loc[medjool_indexes, "class_code"] = 1
# print(df["class_code"])

outcome_df = df["class_code"]
feature_df = df.drop(["class", "class_code", "color"], axis=1)

scaler = MinMaxScaler()
feature_df = scaler.fit_transform(feature_df)

train_features, test_features, train_outcomes, test_outcomes = train_test_split(feature_df, outcome_df, test_size=0.1)

svm = SVC()
svm.fit(train_features, train_outcomes)
test_acc_score = svm.score(test_features, test_outcomes)
train_acc_score = svm.score(train_features, train_outcomes)

print(f"Train Accuracy: {train_acc_score}")
print(f"Test Accuracy: {test_acc_score}")


# Test Features
print(test_features)
predictions = svm.predict(test_features)
print("Predictions")
print(predictions)
print("Decision Function") # How close each sample is to the decision boundary
decision_fxn = svm.decision_function(test_features)
print(decision_fxn)