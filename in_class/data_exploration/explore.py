import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Read the dataset into the "wine_df" variable
wine_df = pd.read_csv("winequality-white.csv", sep=";")
# #Get an overview of the data
# print(wine_df.head())
# #Print the columns - method 1
# print("Columns Index")
# print(wine_df.columns)
# #Print the columns - method 2
# print("Columns as a List")
# print(list(wine_df.columns))
# #Describe the data
# print("Describe the data")
# print(wine_df.describe())


#Indexing
#Return a series with the 'quality' column of of the dataset
# quality_series = wine_df['quality']
# print("Quality series")
# print(quality_series)
# #Get a list of values from a column
# quality_list = wine_df['quality'].values.tolist()
# print("Quality List")
# print(quality_list)

# loc selects column by name, row by idx, includes start and end
# iloc selects both by idx, doesn't include end
# order row, col for both



#
# USING LOC
#

# print("Entire Quality Column")
# print(wine_df.loc[:,'quality'])

# print('First and up to 5th-indexed quality values')
# print(wine_df.loc[0:5,'quality'])

# print('Everything up to 5th indexed row quality value')
# print(wine_df.loc[:5,'quality'])

# print("Everything after 4890, quality")
# print(wine_df.loc[4890:,'quality'])

# print("First through 5th index quality and sulphates")
# print(wine_df.loc[0:5,['quality','sulphates']])



#
# USING ILOC
#

# print("Row indexes 1-2 and column indexes 1-4")
# print(wine_df.iloc[0:3,0:5]) # Note: iloc excludes the last one

# print('4th row index, first item')
# print(wine_df.iloc[4,0])

# print('All columns, row index 4')
# print(wine_df.iloc[4,:])


###Selecting a row or rows based on consitiona
# print('All rows where quality > 8')
# print(wine_df.loc[wine_df['quality'] > 8])
# print('How many rows have a quality > 8?')
# print(len(wine_df.loc[wine_df['quality'] > 8].index))


# # Logical and: &
# print('All rows where quality > 5 AND sulphates > 0.45')
# sub_df=wine_df.loc[(wine_df['quality'] > 5) & (wine_df['sulphates'] > 0.45)]
# print(sub_df.head)
# print('How many rows (wines) is this?')
# print(len(sub_df.index))

# # Logical or: |
# print('All rows where quality > 5 OR sulphates > 0.45')
# sub_df=wine_df.loc[(wine_df['quality'] > 5) | (wine_df['sulphates'] > 0.45)]
# print(sub_df.head)
# print('How many rows (wines) is this?')
# print(len(sub_df.index))


# Find and handle NaNs
# print(wine_df.isna())  # creates a new dataframe with just true or false dependent on if the value at that index is a number
# print(wine_df.isna().value_counts())

# print(wine_df.loc[0,'quality'])
# new_df = wine_df    # Pandas automatically makes a copy, we don't have to specify
# new_df.loc[0,'quality'] = None
# print(new_df.loc[0,'quality'])
# # drop the NaN value
# print("Number of rows before dropping NaN value")
# print(len(new_df.index))
# # Create the new "cleaned" df
# cleaned_df = new_df.dropna()

# # Note this gets rid of that index altogether... so it doesn't exist at all..
# # SO we need to reset the index one of two ways
# # cleaned_df=cleaned_df.reset_index()
# # OR
# # cleaned_df.reset_index(inplace=True)

# print("Number of rows after dropping NaN")
# print(len(cleaned_df.index))



#    VISUALIZATION

# Correlation Heatmap
# print(wine_df.corr())
plot = sns.heatmap(wine_df.corr())
# plt.show()
plt.savefig("correlation_plot.png")
plt.clf()

# Scatterplot
dependent_var = 'quality'
independent_var = 'alcohol'
plt.scatter(wine_df['alcohol'],wine_df['quality'])
plt.title(f'Correlation between {independent_var} and {dependent_var}')
plt.xlabel(independent_var)
plt.ylabel(dependent_var)
plt.savefig("scatter_reversed.png")
plt.clf()