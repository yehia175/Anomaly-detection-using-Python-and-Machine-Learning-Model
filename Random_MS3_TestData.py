import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

trainData = pd.read_csv('Train_data.csv')
testData = pd.read_csv('Test_data.csv')
df = pd.DataFrame(trainData)
test_df = pd.DataFrame(testData)

#Task 1

train_data, test_data = train_test_split(trainData, test_size=0.3, random_state=42)
# Separate the features and labels for the training set
X_train = train_data.loc[:, train_data.columns != 'class']
y_train = (train_data['class'] == 'anomaly').astype(int)

# Separate the features and labels for the testing set
X_test = test_data.loc[:, test_data.columns != 'class']
y_test = (test_data['class'] == 'anomaly').astype(int)

#For the new test set
X_test_new = test_df.loc[:, test_df.columns != 'class']
y_test_new = (test_df['class'] == 'anomaly').astype(int)

# Select only numerical columns, excluding the 'class' column
numerical_columns = df.select_dtypes(include=[np.number]).columns
numerical_columns = [col for col in numerical_columns if col != 'class']
numerical_columns = [col for col in numerical_columns if col != 'hot']  #hot couldn't fit to any of the distributions


def calculate_mse(empirical_counts, fitted_pdf):
    """Calculate Mean Squared Error (MSE) between empirical data and fitted PDF."""
    return np.mean((empirical_counts - fitted_pdf) ** 2)

def best_fit_distribution(df3, flag, unique_values_threshold=10):
    # Define a list of distributions to test
    distributions = [
        stats.alpha, stats.norm, stats.expon, stats.gamma, stats.pareto, stats.beta, stats.lognorm, stats.weibull_min,
        stats.weibull_max, stats.t, stats.f, stats.chi2, stats.gumbel_r, stats.gumbel_l, stats.dweibull,
        stats.genextreme, stats.uniform, stats.arcsine, stats.cosine, stats.exponnorm, stats.foldcauchy
    ]

    if flag == 'original':
        condition_data = df3
    elif flag == 'normal':
        condition_data = df3[df3['class'] == 'normal']
    elif flag == 'anomaly':
        condition_data = df3[df3['class'] == 'anomaly']
    
    pdf_results = {}
    for column in numerical_columns:
        # Loop through each numerical column
        data_conditioned = condition_data[column].dropna()
        if df3[column].nunique() < unique_values_threshold:
            print(f"Skipping '{column}' when based on '{flag}' due to low unique values.")
            continue

        # Calculate IQR and determine the bounds for x-axis
        q1, q3 = np.percentile(df3[column].dropna(), [25, 75])
        iqr = q3 - q1
        lower_bound, upper_bound = (
            np.percentile(df3[column].dropna(), [2, 98])
            if df3[column].max() > q3 + 10 * iqr or df3[column].min() < q1 - 10 * iqr
            else (df3[column].min(), df3[column].max())
        )
        bin_edges = np.linspace(lower_bound, upper_bound, 15)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
        best_mse = float('inf')
        best_distribution = None
        best_params = None
        # Calculate empirical counts based on bin centers
        empirical_counts, _ = np.histogram(data_conditioned, bins=len(bin_centers), range=(bin_centers.min(), bin_centers.max()), density=True)

        for distribution in distributions:
            try:
                # Fit the distribution to data
                params = distribution.fit(data_conditioned)
            
                # Calculate the PDF with fitted parameters
                fitted_pdf = distribution.pdf(bin_centers, *params)

                # Check shapes before calculating MSE
                if len(empirical_counts) != len(fitted_pdf):
                    continue

                # Calculate MSE
                mse = calculate_mse(empirical_counts, fitted_pdf)

                # Update the best distribution if this one has the lowest MSE
                if mse < best_mse:
                    best_mse = mse
                    best_distribution = distribution
                    best_params = params
            except Exception as e:
                continue
        pdf_results[column] = (best_distribution,best_params)

    return pdf_results

#Get the PDFs to be used in naive bayes

pdf_results_anomaly = best_fit_distribution(train_data,'anomaly')
pdf_results_normal = best_fit_distribution(train_data,'normal')
pdf_results_original = best_fit_distribution(train_data,'original')

# Get the PMFs to be used in naive bayes

pmfs_anomaly = {}
pmfs_normal = {}
pmfs_original = {}
categorical_columns = df.select_dtypes(exclude=[np.number]).columns.union(df.columns[df.nunique() < 10])
categorical_columns = categorical_columns.drop('class')
categorical_columns = categorical_columns.union(pd.Index(['hot'])) #Consider hot a categorical column (create a PMF) because it didn't fit to any PDF distribution

for column in categorical_columns: 
    pmf_anomaly = X_train.loc[y_train == 1, column].value_counts(normalize=True).to_dict()
    pmf_normal = X_train.loc[y_train == 0, column].value_counts(normalize=True).to_dict()
    pmf_original = X_train[column].value_counts(normalize=True).to_dict()

    pmfs_anomaly[column] = pmf_anomaly
    pmfs_normal[column] = pmf_normal
    pmfs_original[column] = pmf_original

#Naive bayes estimation
def naive_bayes_estimation(row, conditional_prob_anomaly, conditional_prob_normal):
    prob_anomaly = conditional_prob_anomaly
    prob_normal = conditional_prob_normal
    # Numerical columns
    for column in pdf_results_anomaly:
        best_dist_anomaly, params_anomaly = pdf_results_anomaly[column]
        prob_anomaly *= best_dist_anomaly.pdf(row[column], *params_anomaly)
    for column in pdf_results_normal:
        best_dist_normal, params_normal = pdf_results_normal[column]
        prob_normal *= best_dist_normal.pdf(row[column], *params_normal)
    for column in pdf_results_original:
        best_dist_original, params_original = pdf_results_original[column]
        prob_anomaly /= best_dist_original.pdf(row[column], *params_original)
        prob_normal /= best_dist_original.pdf(row[column], *params_original)
    # Categorical columns
    for column in categorical_columns:
        prob_anomaly *= pmfs_anomaly[column].get(row[column], 1e-6)
        prob_normal *= pmfs_normal[column].get(row[column], 1e-6)
        prob_anomaly /= pmfs_original[column].get(row[column], 1e-6)
        prob_normal /= pmfs_original[column].get(row[column], 1e-6)

    return prob_anomaly, prob_normal
    
conditional_prob_anomaly = len(df[df['class'] == 'anomaly']) / len(df)
conditional_prob_normal = len(df[df['class'] == 'normal']) / len(df)

# Evaluate on the test data
TP, TN, FP, FN = 0, 0, 0, 0
expected_values = []

for (index, row), actual_value in zip(X_test_new.iterrows(), y_test_new):
    naive_prob_anomaly, naive_prob_normal = naive_bayes_estimation(row, conditional_prob_anomaly,conditional_prob_normal)
    if naive_prob_anomaly > naive_prob_normal:
        expected_value = 1
    else:
        expected_value = 0
    expected_values.append(expected_value)
    if expected_value == 1 and actual_value == 1:
        TP += 1
    elif expected_value == 0 and actual_value == 0:
        TN += 1
    elif expected_value == 1 and actual_value == 0:
        FP += 1
    elif expected_value == 0 and actual_value == 1:
        FN += 1

# Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

#Task 2

df_encoded = pd.get_dummies(df.drop(columns=['class']), drop_first=True)

features = df_encoded  # Features after one-hot encoding
target = df['class']  # Target variable from the original DataFrame

#Split the encoded data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

models = {
    'Gaussian Naive Bayes': GaussianNB(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'Bernoulli Naive Bayes': BernoulliNB()
}

metrics_list = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary', pos_label='anomaly') 
    recall = recall_score(y_test, predictions, average='binary', pos_label='anomaly') 
    
    metrics_list.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall
    })

# Convert metrics list to a DataFrame for better visualization
metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)

#Determine the best model
best_model = metrics_df.loc[metrics_df[['Precision', 'Recall']].sum(axis=1).idxmax()]
print(f"\nThe best model is: {best_model['Model']}")

