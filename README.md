# AI_Fundamental_Concepts

Author: Faraz Yusuf Khan

# Importing Necessary Libraries 

import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import Pandas for data manipulation
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Import Seaborn for statistical data visualization
from sklearn import linear_model  # Import linear_model from scikit-learn for linear regression
import mpl_toolkits  # Import mpl_toolkits for additional tools in Matplotlib
import warnings  # Import the warnings module to handle warnings in the code
warnings.filterwarnings("ignore")  # Ignore warnings to enhance code readability

from sklearn.model_selection import train_test_split  # Import train_test_split for splitting the dataset
from sklearn.linear_model import LinearRegression  # Import LinearRegression for linear regression modeling
from sklearn.metrics import mean_squared_error, r2_score  # Import metrics for model evaluation

from matplotlib.cm import viridis  # Import the 'viridis' colormap from Matplotlib
import numpy as np  # Import NumPy for numerical operations
from scipy.stats import norm  # Import norm from scipy.stats for statistical functions

###############################
#   Initial Data Exploration  #
###############################

# Display the contents of the dataset
df

# Check for missing values in the dataset and print the count of missing values for each column
missing_values = df.isnull().sum()
print(missing_values)

# Analyze the correlation between features and the target variable (House Price)
# Select numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr = numeric_cols.corr()

# Display the top ten correlated features with House Price
print('Top ten Correlated Features with House Price:')
print(corr['price'].sort_values(ascending=False)[:10])
print('\n')

# Create the scatter plot with 'viridis' colormap
plt.scatter(x=df['sqft_living'], y=df['price'], c=df['price'], cmap=viridis)

# Add a colorbar for reference
plt.colorbar(label='House Price')

# Set axis labels
plt.ylabel('sqft_living')
plt.xlabel('House Price')

# Save the scatter plot as an image (e.g., PNG)
plt.savefig('scatter_plot_sqft_living.png', dpi=300)

# Show the scatter plot
plt.show()

###############################
#     Training and Testing    #
###############################

# Assigning target and predictor variables to the model
y =  df.loc[:, ['price']]  #target variable

X = df.iloc[:, [3]].values #predictor variable 

# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
random_state = 0)
# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)

# Create a single figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Define color-blind-friendly colors
cb_color1 = 'tab:red'
cb_color2 = 'tab:blue'
cb_color3 = 'tab:green'

# Plot the training set results in the first subplot
ax1.scatter(X_train, y_train, color=cb_color1, marker='o', label='Actual Data (Training)', alpha=0.7)
ax1.plot(X_train, regr.predict(X_train), color=cb_color2, linestyle='--', label='Predicted Data (Training)', linewidth=2)
ax1.set_title('House Price vs sqft_living (Training set)')
ax1.set_xlabel('House Price')
ax1.set_ylabel('sqft_living')
ax1.legend(loc='best')

# Plot the test set results in the second subplot
ax2.scatter(X_test, y_test, color=cb_color1, marker='s', label='Actual Data (Test)', alpha=0.7)
ax2.plot(X_test, regr.predict(X_test), color=cb_color3, linestyle='--', label='Predicted Data (Test)', linewidth=2)
ax2.set_title('House Price vs sqft_living (Test set)')
ax2.set_xlabel('House Price')
ax2.set_ylabel('sqft_living')
ax2.legend(loc='best')

# Adjust layout for subplots
plt.tight_layout()

# Save the combined results plot
plt.savefig('LR_combined_plot_side_by_side_colorblind.png', dpi=300) 

# Display the saved plot
plt.show()

###############################
#            Results          #
###############################

# Print metrics for Simple Linear Regression
print("Simple Linear Regression Metrics")
# Display the coefficients of the linear regression model
print('Coefficients: ', regr.coef_)
# Display the intercept of the linear regression model
print('Intercept: ', regr.intercept_)
# Calculate and display the mean squared error
mse = mean_squared_error(y_test, regr.predict(X_test))
print('Mean squared error: %.8f' % mse)
# Calculate and display the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print('Root Mean squared error: %.8f' % rmse)
# Calculate and display the R^2 value (Coefficient of determination)
r2 = r2_score(y_test, regr.predict(X_test))
print('Coefficient of determination (R^2): %.2f' % r2)
# Predict a single value and display the result
print('Predict single value: ', regr.predict(np.array([[1]])))

##################################
### MULTIPLE LINEAR REGRESSION ###
##################################

# Create the scatter plot with 'viridis' colormap
plt.scatter(x=df['sqft_above'], y=df['price'], c=df['price'], cmap='viridis')
# Add a colorbar for reference
plt.colorbar(label='House Price')
# Set axis labels
plt.ylabel('sqft_above')
plt.xlabel('House Price')
# Save the scatter plot as an image (e.g., PNG)
plt.savefig('scatter_plot_sqft.png', dpi=300)
# Show the  scatter plot
plt.show()

# Create the scatter plot with 'viridis' colormap
plt.scatter(x=df['bathrooms'], y=df['price'], c=df['price'], cmap='viridis')
# Add a colorbar for reference
plt.colorbar(label='House Price')
# Set axis labels
plt.ylabel('bathrooms')
plt.xlabel('House Price')
# Save the scatter plot as an image (e.g., PNG)
plt.savefig('scatter_plot_sqft.png', dpi=300)
# Show the scatter plot
plt.show()

# Create a copy of the original DataFrame
cleaned_df = df.copy()

# Remove outliers from the copied DataFrame
cleaned_df = cleaned_df[cleaned_df['sqft_living'] < 8000]
cleaned_df = cleaned_df[cleaned_df['sqft_above'] < 8000]
cleaned_df = cleaned_df[cleaned_df['bathrooms'] < 8000]

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scatter plot for sqft_living vs. price
scatter1 = axes[0].scatter(x=cleaned_df['sqft_living'], y=cleaned_df['price'], c=cleaned_df['price'], cmap='viridis')
axes[0].set_xlabel('sqft_living')
axes[0].set_ylabel('House Price')
axes[0].set_title('Scatter Plot: sqft_living vs. House Price')

# Scatter plot for sqft_above vs. price
scatter2 = axes[1].scatter(x=cleaned_df['sqft_above'], y=cleaned_df['price'], c=cleaned_df['price'], cmap='viridis')
axes[1].set_xlabel('sqft_above')
axes[1].set_ylabel('House Price')
axes[1].set_title('Scatter Plot: sqft_above vs. House Price')

# Scatter plot for bathrooms vs. price
scatter3 = axes[2].scatter(x=cleaned_df['bathrooms'], y=cleaned_df['price'], c=cleaned_df['price'], cmap='viridis')
axes[2].set_xlabel('bathrooms')
axes[2].set_ylabel('House Price')
axes[2].set_title('Scatter Plot: Bathrooms vs. House Price')

# Adjust layout for subplots
plt.tight_layout()

# Save the combined scatter plot
plt.savefig('combined_scatter_plot.png', dpi=300)

# Display the combined scatter plot
plt.show()

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot histograms for 'sqft_living', 'sqft_above', 'bathrooms', and 'price'
columns = ['sqft_living', 'sqft_above', 'bathrooms', 'price']
colors = ['skyblue', 'salmon', 'lightgreen', 'lightcoral']

for i, column in enumerate(columns):
    ax = axes[i // 2, i % 2]
    
    # Plot histogram
    n, bins, patches = ax.hist(df[column], bins=20, color=colors[i], edgecolor='black', density=True)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

    # Calculate mean and standard deviation for the column
    mu, std = norm.fit(df[column])

    # Create a range of x values for the normal distribution curve
    x = np.linspace(min(bins), max(bins), 100)
    pdf = norm.pdf(x, mu, std)

    # Plot the normal distribution curve
    ax.plot(x, pdf, 'r-', lw=2, label='Normal Distribution')

    # Add legend
    ax.legend()

# Adjust layout for subplots
plt.tight_layout()

# Save the combined histogram plot
plt.savefig('histograms_with_distribution.png', dpi=300)

# Show the histograms
plt.show()

# Apply the logarithmic transformation to the 'price' variable
df['price_log'] = np.log(df['price'])

# Create a figure for the histogram of 'price_log'
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the histogram for 'price_log'
n, bins, patches = ax.hist(df['price_log'], bins=20, color='lightcoral', edgecolor='black', density=True)
ax.set_title('Histogram of log(price)')
ax.set_xlabel('log(price)')
ax.set_ylabel('Frequency')

# Calculate mean and standard deviation for 'price_log'
mu, std = norm.fit(df['price_log'])

# Create a range of x values for the normal distribution curve
x = np.linspace(min(bins), max(bins), 100)
pdf = norm.pdf(x, mu, std)

# Plot the normal distribution curve
ax.plot(x, pdf, 'r-', lw=2, label='Normal Distribution')

# Add legend
ax.legend()

# Adjust layout for the subplot
plt.tight_layout()

# Save the histogram plot for 'price_log'
plt.savefig('histogram_price_log.png', dpi=300)

# Show the histogram
plt.show()

#ASIGNING TARGET AND PREDICTOR VARIABLES
X = cleaned_df.iloc[:, [2,3,10]].values
y =  cleaned_df.loc[:, ['price']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)
model.score(X_test, y_test)
X_test[:1]
model.predict(X_test[:1])

print("Multiple Linear Regression Metrics")
print('Coefficients: ', lr.coef_)
# The intercept
print('Intercept: ', lr.intercept_)
# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, lr.predict(X_test)))
# The Root mean squared error
print('Root Mean squared error: %.8f'
% mean_squared_error(y_test, lr.predict(X_test), squared=False))
# The R^2 value:
print('Coefficient of determination: %.2f'
% r2_score(y_test, lr.predict(X_test)))


# Set up a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for three features against the actual prices (y_test)
scatter_actual = ax.scatter(X_test[:, 0], X_test[:, 1], y_test.flatten(), color='blue', label='Actual Prices', marker='o')
# Add predictions to the scatter plot
scatter_predictions = ax.scatter(X_test[:, 0], X_test[:, 1], predictions.flatten(), color='red', label='Predictions', marker='^')

# Set axis labels
ax.set_xlabel(df.columns[2])
ax.set_ylabel(df.columns[3])
ax.set_zlabel(df.columns[10])

# Set the title
ax.set_title('3D Scatter Plot: Features vs. House Price')

# Set additional 3D plot parameters
ax.azim = -70
ax.dist = 10
ax.elev = 10

# Add a legend
ax.legend()

# Show the 3D scatter plot
plt.show()

# Create a figure for the ColorBlind Friendly 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the actual data points
actual_data = ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c=y_test, cmap=viridis, label='Actual Data')
actual_data.set_edgecolor('k')  # Add a black edge for contrast

# Plot the predicted data points
predicted = model.predict(X_test)
predicted_data = ax.scatter(X_test[:, 0], X_test[:, 1], predicted, c=predicted, cmap=viridis, marker='^', label='Predicted Data')
predicted_data.set_edgecolor('k')  # Add a black edge for contrast

# Customize the plot
ax.set_xlabel('sqft_above')
ax.set_ylabel('bathrooms')
ax.set_zlabel('Price')
ax.set_title('Multiple Linear Regression Model')

# Add a color bar for reference
cbar = fig.colorbar(predicted_data, ax=ax)
cbar.set_label('Price')

# Add a legend
ax.legend()
# Save the plot as an image
plt.savefig('multiple_linear_regression_3D_plot.png', dpi=300)

# Show the 3D scatter plot
plt.show()

