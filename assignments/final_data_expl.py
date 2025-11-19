import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("../datasets/personal_finance_tracker_dataset.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# print(df.shape)
# print(df.head)
# print(df.describe())

# print(df.nunique()) 
# print(df.dtypes)

### Data Cleaning and Preprocessing

# Converting the date column to a datetime object
# try:
#    df['date'] = pd.to_datetime(df['date'])
#    print('Converted date column to datetime successfully.')
# except Exception as e:
#    print('Error converting date column:', e)  # This error handling is crucial for other notebook creators facing similar issues

# Checking for missing values
# missing_values = df.isnull().sum()
# print('Missing values in each column:')
# print(missing_values)
# print(missing_values[missing_values > 0])


# print(df.describe())
# print(df.dtypes)

# numeric_df = df.select_dtypes(include=[np.number])
# # plt.figure(figsize=(14,12))
# plot = sns.heatmap(numeric_df.corr(),cmap="coolwarm")
# plt.xticks(rotation=65, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# # plt.figure()
# plt.savefig("A4_visualizations/numeric_vals_correlation_plot.png", bbox_inches='tight')
# plt.clf()


# Scatterplot
# dependent_var = 'financial_scenario'
# independent_var = 'date'
# plt.scatter(df[independent_var],df[dependent_var])
# plt.title(f'Correlation between {independent_var} and {dependent_var}')
# plt.xlabel(independent_var)
# plt.ylabel(dependent_var)
# plt.savefig(f"A4_visualizations/scatter_{independent_var}_{dependent_var}.png")
# plt.clf()


##Selecting a row or rows based on consitiona
# filtered = df.loc[df['actual_savings'] > 6000, [
#    'user_id',
#    'monthly_income',
#    'monthly_expense_total',
#    'loan_payment',
#    'investment_amount',
#    'emergency_fund',
#    'discretionary_spending',
#    'essential_spending',
#    'rent_or_mortgage',
#    'actual_savings'
# ]]
# print('All rows where savings > 6000 (showing selected columns):')
# print(filtered)

# print('All rows with fraud')
# print(df.loc[df['fraud_flag'] > 0])
# print('How many rows have fraud?')
# print(len(df.loc[df['fraud_flag'] > 0].index))

# Scatterplot
# dependent_var = 'financial_stress_level'
# independent_var = 'monthly_income'
# plt.scatter(df[independent_var],df[dependent_var])
# plt.title(f'Correlation between {independent_var} and {dependent_var}')
# plt.xlabel(independent_var)
# plt.ylabel(dependent_var)
# plt.savefig(f"A4_visualizations/scat_{independent_var}_{dependent_var}.png")
# plt.clf()

# var = 'monthly_income'
# plt.figure(figsize=(8, 6))
# sns.histplot(df[var], kde=True, bins=30)#, color='teal')
# plt.title(f"Distribution of {var}")
# plt.xlabel(f"{var}")
# plt.ylabel("Frequency")
# plt.savefig(f"A4_visualizations/dist_{var}.png")
# plt.clf()


# Logical or: |
# print('All rows where income > 6800 OR income < 1200')
# sub_df=df.loc[(df['monthly_income'] > 6800.0) | (df['monthly_income'] <1200.0)]
# print(sub_df.head)
# print('How many rows (entries) is this?')
# print(len(sub_df.index))

# df_grouped = df.groupby('date').mean(numeric_only=True)
# over_time_1 = 'actual_savings'
# over_time_2 = 'budget_goal'
# # over_time_3 = 'savings_goal'
# plt.figure(figsize=(12, 6))
# plt.plot(df_grouped.index, df_grouped[over_time_1], label='Actual Savings')
# plt.plot(df_grouped.index, df_grouped[over_time_2], label='Budget Goal', linestyle='--')
# plt.title("Averaged Savings and Budget Goals Over Time")
# plt.xlabel("Date")
# plt.ylabel("Amount ($)")
# plt.legend()
# plt.savefig(f"A4_visualizations/savings_over_time.png")
# plt.clf()


# # Test and Training set
# dep='actual_savings'
# indep='monthly_income'
# training_set, testing_set = train_test_split(df, test_size=0.2)
# # Repeat the model training process with training/test set
# linear_regression_model_2 = LinearRegression()
# linear_regression_model_2.fit(training_set[[indep]],training_set[[dep]])
# model_m = linear_regression_model_2.coef_[0][0]
# model_b = linear_regression_model_2.intercept_[0]
# print(f'Training Set Model Regression Equation: y = {model_m}x + {model_b}')

# # Get the Coefficient of Determination
# score = linear_regression_model_2.score(testing_set[[indep]],testing_set[[dep]])
# print(f'Coefficient of determination for test set: {score}')


# plt.figure(figsize=(8, 5))
# sns.countplot(data=df[df['fraud_flag'] == 1], x='financial_scenario')
# plt.title("Fraud Cases by Financial Scenario")
# plt.xlabel("Scenario")
# plt.ylabel("Count of Fraudulent Transactions")
# plt.savefig(f"A4_visualizations/fraud_for_scenario.png")
# plt.clf()



# ----------
# Data Preprocessing
# ----------
# Features: drop everything but the fraud flag column
non_features=['fraud_flag','date']
print(f"Shape of original data: {df.shape}")

features_old = df.drop(non_features, axis=1)
print(f"Shape of the original Features dataframe: {features_old.shape}")
# print(features_old.head())
# print(features_old.dtypes)
# Replace string categorical variables (financial_scenario, income_type, cash_flow_status, and financial_stress_level) with unique columns
# small_set = df[['financial_scenario', 'income_type', 'cash_flow_status', 'financial_stress_level']]
# print(small_set.describe())
features=pd.get_dummies(features_old, columns=['financial_scenario', 'income_type', 'cash_flow_status', 'financial_stress_level', 'category'])
# print(features.dtypes)
print(f"Shape of the updated Features dataframe: {features.shape}")

# Outcomes: just the fraud
outcomes = df['fraud_flag']
print(f"Shape of the Outcomes dataframe: {outcomes.shape}")
# print(outcomes.head())

# Normalize everything
scaler = MinMaxScaler()
# Fit both because model expects them to be in the same format (scaler converts to numpy array)
features = scaler.fit_transform(features)
outcomes = scaler.fit_transform(outcomes.values.reshape(-1, 1))
# Split into training and test sets
train_x, test_x, train_y, test_y = train_test_split(features, outcomes, test_size=0.2)
print(f"Normalized train feature shape: {train_x.shape}")
print(f"Normalized train outcome shape: {train_y.shape}")
print(f"Normalized test feature shape: {test_x.shape}")
print(f"Normalized test outcome shape: {test_y.shape}")
