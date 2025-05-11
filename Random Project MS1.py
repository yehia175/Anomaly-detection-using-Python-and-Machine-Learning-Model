# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 19:19:54 2024

@author: G5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway


# Load the data
data = pd.read_csv('Train_data1.csv')

# a. List the data fields (columns)
print("Data fields (columns):")
print(data.columns.tolist())

# b. List the types of the data fields (columns)
print("Data types of each field:")
print(data.dtypes)

# c. Check for missing or infinite data
missing_data = data.isnull().sum()

print("Missing Data in Each Column:")
print(missing_data[missing_data > 0])

# Check for infinite values
infinite_data = data.isin([float('inf'), float('-inf')]).sum()

print("\nInfinite Data in Each Column:")
print(infinite_data[infinite_data > 0])

# d. Number of categories in each field (column)
category_counts = data.nunique() #d
print("Number of unique categories in each column:")
print(category_counts)

# e. Maximum, Minimum, Average, and Variance of each data field
print("Statistics for each numerical data field:")
stats = data.describe().T[['min', 'max', 'mean', 'std']]
stats['variance'] = stats['std']**2 #calculates variance by squaring standard deviation
print(stats[['min', 'max', 'mean', 'variance']])

# f. Divide each data field into 4 quarters and check statistics
print("Statistics for each quarter of numerical data fields:")
for column in data.select_dtypes(include=[np.number]).columns:
    # Define quarters
    q1 = data[column].quantile(0.25)
    q2 = data[column].quantile(0.50)
    q3 = data[column].quantile(0.75)

    quarters = {
        'Q1': data[data[column] <= q1],
        'Q2': data[(data[column] > q1) & (data[column] <= q2)],
        'Q3': data[(data[column] > q2) & (data[column] <= q3)],
        'Q4': data[data[column] > q3],
    }

    print(f"\nStatistics for '{column}' by quarter:")
    for key, subset in quarters.items():
        max_value = subset[column].max()
        min_value = subset[column].min()
        mean_value = subset[column].mean()
        variance_value = subset[column].var()

        print(f"{key} - Max: {max_value}, Min: {min_value}, Mean: {mean_value}, Variance: {variance_value}")

#3

for column in data.columns:
    valid_data = data[column].dropna()  # Remove NaN values

    # Check if the column is discrete or continuous
    if data[column].dtype == 'object' or len(data[column].unique()) < 20:  # Discrete data (categorical or few unique values)
        # Calculate PMF (Probability Mass Function)
        pmf_values = valid_data.value_counts(normalize=True)
        
        # Plot PMF
        plt.figure(figsize=(8, 6))
        pmf_values.plot(kind='bar', color='blue', alpha=0.7)
        plt.title(f'PMF of {column}')
        plt.xlabel(column)
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.show()
        
    else:  # Continuous data (numeric with many unique values)
        # Calculate PDF (Probability Density Function) using kernel density estimation (KDE)
       def calculate_pdf_smooth(data, bins=30, smooth_factor=10):
 
           hist, bin_edges = np.histogram(data, bins=bins, density=True)
           bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Find the centers of the bins
    
    # Smooth the histogram using a moving average (simple smoothing)
           pdf_smooth = np.convolve(hist, np.ones(smooth_factor)/smooth_factor, mode='same')
    
           return bin_centers, pdf_smooth
         
#4

# Function to calculate CDF from the PDF
def calculate_cdf(pdf):

    return np.cumsum(pdf) * (bin_centers[1] - bin_centers[0])  # Multiply by bin width for normalization

# For each numerical column, calculate and plot PDF and CDF as curves
for column in data.columns:
    valid_data = data[column].dropna()  # Remove NaN values
    
    if data[column].dtype in ['int64', 'float64']:  # For numerical data
        # Calculate smoothed PDF
        bin_centers, pdf_smooth = calculate_pdf_smooth(valid_data, bins=30, smooth_factor=15)

        # Calculate CDF from PDF
        cdf = calculate_cdf(pdf_smooth)
        
        # Plot PDF as a curve (smooth)
        plt.figure(figsize=(12, 6))
        plt.plot(bin_centers, pdf_smooth, color='blue', lw=2, label='PDF (Smoothed)')
        plt.title(f'PDF of {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot CDF as a curve
        plt.figure(figsize=(12, 6))
        plt.plot(bin_centers, cdf, color='red', lw=2, label='CDF')
        plt.title(f'CDF of {column}')
        plt.xlabel(column)
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    else:  # For non-numerical (categorical) data
        # Calculate CDF for categorical data
        cdf_values = valid_data.value_counts(normalize=True).sort_index().cumsum()

        # Create an index for the categories
        category_index = cdf_values.index

        # Plot CDF as a line
        plt.figure(figsize=(12, 6))
        plt.plot(category_index, cdf_values, color='red', lw=2, marker='o', label='CDF')
        plt.title(f'CDF of {column}')
        plt.xlabel(column)
        plt.ylabel('Cumulative Probability')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()

#5

def plot_pmf_pdf_with_condition(df, condition_col=None, condition_value=None):
    for col in df.columns:
        plt.figure(figsize=(10, 6))

        # Remove invalid or NaN values from the column
        valid_data = df[col].dropna()  # to remove NaN values

        # Original PMF or PDF
        if valid_data.dtype in ['int64', 'float64']:
            if valid_data.nunique() < 20:  # Discrete
                pmf = valid_data.value_counts(normalize=True)
                plt.bar(pmf.index, pmf.values, width=0.5, color='blue', alpha=0.6, label='Original PMF')
                
                # Conditional PMF
                if condition_col and condition_value is not None:
                    conditional_data = df[df[condition_col] == condition_value][col].dropna()
                    conditional_pmf = conditional_data.value_counts(normalize=True)
                    plt.bar(conditional_pmf.index, conditional_pmf.values, width=0.4, color='orange', alpha=0.6, label='Conditional PMF')

                plt.title(f'PMF of {col}')
                plt.xlabel(col)
                plt.ylabel('Probability')
            else:  # Continuous
               
                # Calculate and plot the KDE manually (using Silverman's rule)
                density = np.linspace(min(valid_data), max(valid_data), 100)
                bandwidth = 1.06 * np.std(valid_data) * len(valid_data) ** (-1 / 5.)  # Silverman's rule
                kde = np.sum(np.exp(-(density[:, None] - valid_data.values) ** 2 / (2 * bandwidth ** 2)), axis=1)
                kde /= np.sqrt(2 * np.pi) * bandwidth * len(valid_data)

                plt.plot(density, kde, color='blue', label='Original PDF')

                # Conditional PDF
                if condition_col and condition_value is not None:
                    conditional_data = df[df[condition_col] == condition_value][col].dropna()
                    if len(conditional_data) > 0:
                        conditional_kde = np.sum(np.exp(-(density[:, None] - conditional_data.values) ** 2 / (2 * bandwidth ** 2)), axis=1)
                        conditional_kde /= np.sqrt(2 * np.pi) * bandwidth * len(conditional_data)
                        plt.plot(density, conditional_kde, color='orange', label='Conditional PDF')

                plt.title(f'PDF of {col}')
                plt.xlabel(col)
                plt.ylabel('Density')
        else:  # Categorical data
            pmf = valid_data.value_counts(normalize=True)
            plt.bar(pmf.index, pmf.values, width=0.5, color='blue', alpha=0.6, label='Original PMF')

            # Conditional PMF
            if condition_col and condition_value is not None:
                conditional_data = df[df[condition_col] == condition_value][col].dropna()
                conditional_pmf = conditional_data.value_counts(normalize=True)
                plt.bar(conditional_pmf.index, conditional_pmf.values, width=0.4, color='orange', alpha=0.6, label='Conditional PMF')

            plt.title(f'PMF of {col}')
            plt.xlabel(col)
            plt.ylabel('Probability')

        plt.grid(True)
        plt.legend()
        plt.show()


plot_pmf_pdf_with_condition(data, condition_col='class', condition_value='anomaly')


#6

def scatter_plot(df): 
    
    # Automatically get all column names
    fields = data.columns.tolist()

    # Loop through each pair of consecutive fields
    for i in range(len(fields) - 1):
        col1 = fields[i]
        col2 = fields[i + 1]
        plt.figure(figsize=(8, 6))
        plt.scatter(df[col1], df[col2], alpha=0.5, color='b')
        plt.title(f'Scatter plot between {col1} and {col2}')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        plt.show()

# Call the scatter plot function
# Example: replace 'src_bytes' and 'dst_bytes' with any two fields from your dataset
scatter_plot(data)

#7

def plot_joint_distribution(data):
    # Automatically get all column names
    fields = data.columns.tolist()

    # Loop through each pair of consecutive fields
    for i in range(len(fields) - 1):
        field1 = fields[i]
        field2 = fields[i + 1]
        
        # Check if fields are categorical or numerical
        is_field1_categorical = data[field1].dtype == 'object' or data[field1].nunique() < 20
        is_field2_categorical = data[field2].dtype == 'object' or data[field2].nunique() < 20

        # If both fields are continuous
        if not is_field1_categorical and not is_field2_categorical:
            # Joint PDF as a 2D histogram
            plt.figure(figsize=(10, 6))
            plt.hist2d(data[field1].dropna(), data[field2].dropna(), bins=30, cmap='Blues')
            plt.colorbar(label='Count')
            plt.title(f'Joint PDF of {field1} and {field2} (Numerical)')
            plt.xlabel(field1)
            plt.ylabel(field2)
            plt.show()
        
        # If both fields are discrete
        elif is_field1_categorical and is_field2_categorical:
            contingency_table = pd.crosstab(data[field1], data[field2], normalize='all')
            print(f"Joint PMF (Categorical) for {field1} and {field2}:")
            print(contingency_table)

            # Plotting the joint PMF as a heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(contingency_table, cmap='Blues', aspect='auto')
            plt.colorbar(label='Probability')
            plt.title(f'Joint PMF of {field1} and {field2} (Categorical)')
            plt.xticks(ticks=range(len(contingency_table.columns)), labels=contingency_table.columns, rotation=45)
            plt.yticks(ticks=range(len(contingency_table.index)), labels=contingency_table.index)
            plt.xlabel(field2)
            plt.ylabel(field1)
            plt.tight_layout()
            plt.show()
        
        else:
            print(f"Skipping {field1} and {field2} because one is discrete and the other is continuous.")

plot_joint_distribution(data)


#8

def plot_conditional_joint_distribution(data, attack_type):
    filtered_data = data[data['class'] == 'anomaly']
    
    # Automatically get all column names
    fields = filtered_data.columns.tolist()

    # Loop through each pair of consecutive fields
    for i in range(len(fields) - 1):
        field1 = fields[i]
        field2 = fields[i + 1]

        # Check if fields are categorical or numerical
        is_field1_categorical = filtered_data[field1].dtype == 'object' or filtered_data[field1].nunique() < 20
        is_field2_categorical = filtered_data[field2].dtype == 'object' or filtered_data[field2].nunique() < 20
        
        # Joint PMF for categorical fields
        if is_field1_categorical and is_field2_categorical:
            contingency_table = pd.crosstab(filtered_data[field1], filtered_data[field2], normalize='all')
            print(f"Joint PMF (Categorical) for Attack Type '{attack_type}':")
            print(contingency_table)

            # Plotting the joint PMF as a heatmap
            plt.figure(figsize=(12, 8))
            plt.imshow(contingency_table, cmap='Blues', aspect='auto')
            plt.colorbar(label='Probability')
            plt.title(f'Joint PMF of {field1} and {field2} (Categorical) for Attack Type: {attack_type}')
            plt.xticks(ticks=range(len(contingency_table.columns)), labels=contingency_table.columns, rotation=45)
            plt.yticks(ticks=range(len(contingency_table.index)), labels=contingency_table.index)
            plt.xlabel(field2)
            plt.ylabel(field1)
            plt.tight_layout()
            plt.show()

        # Joint PDF for numerical fields
        elif not is_field1_categorical and not is_field2_categorical:
            # Joint PDF as a 2D histogram
            plt.figure(figsize=(10, 6))
            plt.hist2d(filtered_data[field1].dropna(), filtered_data[field2].dropna(), bins=30, cmap='Blues')
            plt.colorbar(label='Count')
            plt.title(f'Joint PDF of {field1} and {field2} (Numerical) for Attack Type: {attack_type}')
            plt.xlabel(field1)
            plt.ylabel(field2)
            plt.show()

        else:
            print(f"Skipping {field1} and {field2} because one is categorical and the other is numerical.")


plot_conditional_joint_distribution(data, 'class')


#9    #Correlation only for numerical data

df = pd.DataFrame(data)
numeric_df = df.select_dtypes(include=[np.number])
# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Print the correlation matrix
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar(label='Correlation coefficient')
plt.xticks(np.arange(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(np.arange(len(correlation_matrix.index)), correlation_matrix.index)
plt.title('Correlation Matrix')
plt.show()

#10

df = pd.DataFrame(data)
# Loop over all fields except 'attack_type'
for column in df.columns:
    if column != 'class':
        # Check the data type of the field
        if df[column].dtype == 'object' or df[column].dtype == 'category':
            # Chi-Squared Test for categorical variables
            contingency_table = pd.crosstab(df['class'], df[column])
            chi2, p, _, _ = chi2_contingency(contingency_table)
            print(f'Chi-Squared Test for {column}: chi2 = {chi2}, p-value = {p}')
            if p < 0.05:
                print(f"{column} is dependent on the type of attack. \n")
            else:
                print(f"{column} is not dependent on the type of attack. \n")
        else:
            # ANOVA for continuous variables
            attack_groups = [df[df['class'] == attack][column] for attack in df['class'].unique()]
            anova_result = f_oneway(*attack_groups)
            print(f'ANOVA Test for {column}: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}')
            if anova_result.pvalue < 0.05:
                print(f"{column} is dependent on the type of attack. \n")
            else:
                print(f"{column} is not dependent on the type of attack. \n")
 
