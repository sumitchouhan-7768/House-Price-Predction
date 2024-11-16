import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import variation, zscore
import scipy.stats as stats
import pickle
# Import Dataset
df = pd.read_csv(r"C:\Users\HP\.vscode\.vscode\.vscode\Machine Learning\House_data.csv")

# Setting variables 
space = df['sqft_living']
price = df['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fitting simple linear regression to the Training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the prices for the test set
pred = regressor.predict(x_test)

# Visualizing the Training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title("Training Dataset Visualization")
plt.xlabel("Space (sqft_living)")
plt.ylabel("Price")
plt.show()

# Visualizing the Test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue') # use training line for reference
plt.title("Test Dataset Visualization")
plt.xlabel("Space (sqft_living)")
plt.ylabel("Price")
plt.show()

# Statistical summaries
print("Mean of price:", df['price'].mean())
print("Median of price:", df['price'].median())
print("Mode of price:", df['price'].mode()[0])  # Mode returns a series, so take the first value

print("Variance of price:", df['price'].var())
print("Standard deviation of price:", df['price'].std())

# Coefficient of variation (CV)
print("Coefficient of variation (CV) for the price:", variation(df['price']))

# Correlation
print("Correlation between price and sqft_living:", df['price'].corr(df['sqft_living']))

# Skewness
print("Skewness of price:", df['price'].skew())

# Standard error
print("Standard error of price:", df['price'].sem())

# Z-score
z_scores = stats.zscore(df['price'])
print("Z-scores for the price column:", z_scores)

# Degree of Freedom
degree_of_freedom = df.shape[0] - df.shape[1]
print("Degree of Freedom for the dataset:", degree_of_freedom)

# Sum of Squares Regression (SSR)
y_mean = np.mean(y_test)  # Calculate mean of y_test for SSR
SSR = np.sum((pred - y_mean) ** 2)
print("Sum of Squares Regression (SSR):", SSR)

# Sum of Squared Errors (SSE)
SSE = np.sum((y_test - pred) ** 2)
print("Sum of Squared Errors (SSE):", SSE)

# Sum of Squares Total (SST)
SST = np.sum((y_test - y_mean) ** 2)
print("Sum of Squares Total (SST):", SST)

# R-Squared calculation
r_square = 1 - (SSE / SST)
print("R-Squared:", r_square)

# Check model performance
bias = regressor.score(x_train, y_train)
variance = regressor.score(x_test, y_test)
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
test_mse = mean_squared_error(y_test,pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
#save the trained model to disk
filename = 'LR_House_pred.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as LR_House_pred.pkl")    

import os
print(os.getcwd())