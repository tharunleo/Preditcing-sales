# Week 1: Project Kickoff and Dataset Familiarization

## Documentation
"""
Objective: Understand the project scope and dataset. Define success metrics for the predictive model.

Steps:
1. Project Kickoff
   - Review project objectives and deliverables.
   - Set up the development environment with necessary tools and libraries.
2. Dataset Familiarization
   - Load and inspect the dataset.
   - Identify key fields such as Customer ID, Transaction Date, Merchant, Category, Transaction Amount, Location, and optional Labels.
3. Define Success Metrics
   - Define evaluation metrics such as Accuracy, RMSE (Root Mean Squared Error), or MAPE (Mean Absolute Percentage Error).
"""

# Importing necessary libraries
# pandas: For data manipulation and analysis.
# numpy: For numerical operations and handling arrays.
# matplotlib and seaborn: For data visualization to explore trends and patterns visually.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from Kaggle
# Dataset URL: https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset
# Replace 'path_to_dataset.csv' with the actual file path of the downloaded dataset.
file_path = "credit_card_transactions.csv"  # Ensure this file is downloaded and placed in the working directory
data = pd.read_csv(file_path)

# Display basic dataset information
# data.info() provides a concise summary of the dataset, including the number of non-null values and data types for each column.
print("Dataset Info:")
data.info()

# Preview the first few rows of the dataset to understand its structure and contents.
print("\nDataset Preview:")
print(data.head())

# Check for missing values in each column to identify potential issues in the dataset.
# This step ensures that missing data is handled properly during preprocessing.
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Define success metrics
# Metrics like RMSE, MAE, or MAPE help evaluate the performance of predictive models in forecasting sales accurately.
define_metrics = "RMSE, MAE, or MAPE based on business requirements"
print(f"\nDefined Metrics: {define_metrics}")

# Week 2: Data Exploration and Cleaning

## Documentation
"""
Objective: Perform exploratory data analysis (EDA) and clean the dataset for machine learning.

Steps:
1. Exploratory Data Analysis (EDA)
   - Analyze spending patterns across categories.
   - Visualize data trends using histograms, boxplots, and heatmaps.
2. Data Cleaning
   - Handle missing values (e.g., impute or remove them).
   - Remove duplicates.
   - Correct inconsistent data formats (e.g., date parsing).
"""

# Data Exploration
# Visualize the distribution of transaction amounts to understand how spending varies.
plt.figure(figsize=(10, 6))
sns.histplot(data['Transaction Amount'], bins=30, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()

# Analyze spending by category to identify top-performing or frequently used categories.
category_spending = data.groupby('Category')['Transaction Amount'].sum().sort_values(ascending=False)
print("\nSpending by Category:")
print(category_spending)

# Visualize total spending by category using a bar chart.
plt.figure(figsize=(10, 6))
category_spending.plot(kind='bar', color='skyblue')
plt.title("Total Spending by Category")
plt.xlabel("Category")
plt.ylabel("Total Spending")
plt.show()

# Data Cleaning
# Handle missing values by:
# 1. Dropping columns with more than 10% missing values (threshold).
# 2. Filling other missing values using forward fill (ffill) to propagate last valid data.
threshold = 0.1
columns_to_drop = missing_values[missing_values > len(data) * threshold].index
data_cleaned = data.drop(columns=columns_to_drop)
data_cleaned.fillna(method='ffill', inplace=True)

# Remove duplicate rows to ensure data consistency.
data_cleaned = data_cleaned.drop_duplicates()

# Convert 'Transaction Date' to datetime format to facilitate time-based analysis.
data_cleaned['Transaction Date'] = pd.to_datetime(data_cleaned['Transaction Date'], errors='coerce')

# Verify the cleaned dataset by checking its structure and contents again.
print("\nCleaned Dataset Info:")
data_cleaned.info()

# Save the cleaned dataset to a new file for use in subsequent steps.
cleaned_file_path = "cleaned_credit_card_transactions.csv"
data_cleaned.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned dataset saved to {cleaned_file_path}")
