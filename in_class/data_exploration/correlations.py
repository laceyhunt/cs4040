import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load in the dataset
wine_df = pd.read_csv('../winequality-white.csv', sep=';')
plt.scatter(wine_df['alcohol'],wine_df['quality'])
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.title('Alcohol vs Quality')
plt.savefig("alc_qual_scatter.png")
plt.clf()

# Manual Regression between alcohol and quality
# First, get variables we need to make calculations
x = 'alcohol'
y = 'quality'
n = len(wine_df.index)
sum_x = wine_df[x].sum()
sum_y = wine_df[y].sum()
x_mean = sum_x/n
y_mean = sum_y/n
sum_x_times_y = (wine_df[x]*wine_df[y]).sum()
sum_x_squareds = wine_df[x].pow(2).sum()
print(f'n: {n}')
print(f'Sum of x vals: {sum_x}')
print(f'Sum of y vals: {sum_y}')
print(f'Mean x val: {x_mean}')
print(f'Mean y val: {y_mean}')
print(f'Sum of x*y: {sum_x_times_y}')
print(f'Sum of x^2s: {sum_x_squareds}')

# Next, calculate num and denom for m
S_xy = sum_x_times_y - (sum_x*sum_y)/n
S_xx = sum_x_squareds - (sum_x*sum_x)/n
print(f'Sxy: {S_xy}  Sxx: {S_xx}')
# Get our line variables
m = S_xy/S_xx
b = y_mean - m*x_mean
print(f'Regression equation: y = {m}x + {b}')

# Now lets predict a value
row = 5
alcohol_val = wine_df.loc[5, x]
quality_val = wine_df.loc[5, y]
predicted_quality = alcohol_val*m + b
print(f"""For {x} value {alcohol_val}, the predicted {y} was {predicted_quality}
      and the actual {y} is {quality_val}.""")

# Coefficient of determination
wine_df['predictions'] = wine_df[x]*m+b
residual_squares = (wine_df[y]-wine_df['predictions']).pow(2).sum()
total_squares = (wine_df[y] - y_mean).pow(2).sum()
r_squared = 1-residual_squares/total_squares
print(f'Coefficient of determination: {r_squared}')


# Now, lets run the model with scikit learn
linear_regression_model = LinearRegression()
# This can handle multiple regression, so it expects multiple x values... hence the fancy brackets
linear_regression_model.fit(wine_df[[x]],wine_df[[y]]) # note: this call takes a while on older machines
model_m = linear_regression_model.coef_[0][0]
model_b = linear_regression_model.intercept_[0]
print('For the scikit learn Regression Model...')
print(f'Regression Equation: y = {model_m}x + {model_b}')

# Predict the same value
predicted_model_quality = linear_regression_model.predict([[alcohol_val]])
# predicted_model_quality = linear_regression_model.predict([[wine_df.loc[5, x]]])
print(f"""For {x} value {alcohol_val}, the predicted {y} was {predicted_model_quality}
      and the actual {y} is {quality_val}.""")

# Coefficient of determination
score = linear_regression_model.score(wine_df[[x]],wine_df[[y]])
print(f'Coefficient of determination: {score}')

# # Plot the best fit line
# plt.scatter(wine_df[x],wine_df[y])
# plt.plot(wine_df[x],wine_df['predictions'], linestyle='solid')
# plt.xlabel(x)
# plt.ylabel(y)
# plt.title('Alcohol vs. Quality with best fit line')
# plt.savefig("alc_qual_best_fit.png")
# plt.clf()



# Test and Training set
training_set, testing_set = train_test_split(wine_df, test_size=0.1)
# Repeat the model training process with training/test set
linear_regression_model_2 = LinearRegression()
linear_regression_model_2.fit(training_set[['alcohol']],training_set[['quality']])
model_m = linear_regression_model_2.coef_[0][0]
model_b = linear_regression_model_2.intercept_[0]
print(f'Training Set Model Regression Equation: y = {model_m}x + {model_b}')

# Get the Coefficient of Determination
score = linear_regression_model_2.score(testing_set[['alcohol']],testing_set[['quality']])
print(f'Coefficient of determination for test set: {score}')