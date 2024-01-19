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

