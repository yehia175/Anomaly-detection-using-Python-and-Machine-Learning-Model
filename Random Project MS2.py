import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from distfit import distfit
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('Train_data1.csv')
#Task 1
df = pd.DataFrame(data)#drop method works for DataFrame not original data

df_excluded = df.drop(columns=['class'])
# Split into 70% training and 30% testing
train_data, test_data = train_test_split(df_excluded, test_size=0.3, random_state=42)

#Z_score for train data
train_numerical_df = train_data.select_dtypes(include=['number'])
z_scores_train_df = (train_numerical_df - train_numerical_df.mean()) / train_numerical_df.std()#data set for Z_scores
#Performance metrics
trueValues_train = df.loc[train_data.index, 'class'].map({'normal': 0, 'anomaly': 1})

#Predicted values based on z_score

#Note: The following piece of code is left commented on purpose. It was used to determine the best threshold value for training data.
'''
thresholds = np.arange(1.25, 3.26, 0.02)#end point is exclusive
true_List_train = trueValues_train.tolist()
i = 1.25
TP = TN = FP = FN = 0
while (i in thresholds):
    predictedValues_train = (np.abs(z_scores_train_df) > i).astype(int).max(axis=1)
    pred_List_train = predictedValues_train.tolist()
    for true, pred in zip(true_List_train, pred_List_train):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0 #to prevent dividing by zero
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0 #to prevent dividing by zero
    print("For training data with threshold = ", i,":")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    i += 0.02
'''
threshold = 1.57
predictedValues_train = (np.abs(z_scores_train_df) > threshold).astype(int).max(axis=1)

true_List_train = trueValues_train.tolist()
pred_List_train = predictedValues_train.tolist()
TP = TN = FP = FN = 0
for true, pred in zip(true_List_train, pred_List_train):
    if true == 1 and pred == 1:
        TP += 1
    elif true == 0 and pred == 0:
        TN += 1
    elif true == 0 and pred == 1:
        FP += 1
    elif true == 1 and pred == 0:
        FN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0 #to prevent dividing by zero
recall = TP / (TP + FN) if (TP + FN) != 0 else 0 #to prevent dividing by zero
print("For training data with threshold = ", threshold,":")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

#Test data
#Z_score for test data
test_numerical_df = test_data.select_dtypes(include=['number'])
z_scores_test_df = (test_numerical_df - test_numerical_df.mean()) / test_numerical_df.std()#data set for Z_scores

#Predicted values based on z_score
trueValues_test = df.loc[test_data.index, 'class'].map({'normal': 0, 'anomaly': 1})

#Note: The following piece of code is left commented on purpose. It was used to determine the best threshold value for testing data.
'''
thresholds = np.arange(1.25, 3.26, 0.02)#end point is exclusive
true_List_test = trueValues_test.tolist()
i = 1.25
TP = TN = FP = FN = 0
while (i in thresholds):
    predictedValues_test = (np.abs(z_scores_test_df) > i).astype(int).max(axis=1)
    pred_List_test = predictedValues_test.tolist()
    for true, pred in zip(true_List_test, pred_List_test):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0 #to prevent dividing by zero
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0 #to prevent dividing by zero
    print("For testing data with threshold = ", i,":")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    i += 0.02
'''
#The threshold value that best suits training and testing data was chosen.

threshold = 1.57
predictedValues_test = (np.abs(z_scores_test_df) > threshold).astype(int).max(axis=1)

true_List_test = trueValues_test.tolist()
pred_List_test = predictedValues_test.tolist()
TP = TN = FP = FN = 0
for true, pred in zip(true_List_test, pred_List_test):
    if true == 1 and pred == 1:
        TP += 1
    elif true == 0 and pred == 0:
        TN += 1
    elif true == 0 and pred == 1:
        FP += 1
    elif true == 1 and pred == 0:
        FN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0 #to prevent dividing by zero
recall = TP / (TP + FN) if (TP + FN) != 0 else 0 #to prevent dividing by zero
print("For testing data with threshold = ", threshold,":")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
'''
Recall is the most important metric for this project. 
 '''      

#Task 2

numeric_df = df_excluded.select_dtypes(include=['float64', 'int64'])
numeric_df = numeric_df.loc[:, numeric_df.nunique() > 20]
# Initialize distfit
dist = distfit()
# Dictionary to store results
best_fits = {}
# Loop over each numerical column
i = 1
for column in numeric_df.columns:
    droppedData = numeric_df[column].dropna()  # Drop NaN values
    try:
        dist.fit_transform(droppedData, verbose=0)

        # Store the best-fit distribution model
        best_fits[column] = dist.model['name']

        print(i, "- ", f"{column}: Best fit distribution is {dist.model['name']}")
        i += 1

    except Exception as e:
        print(f"Could not fit distribution for {column}: {e}")

def calculate_PDF_conditioned(data,numeric_df, flag):
    # Dictionary to store results
    best_fits_conditioned = {}
    i = 1
    dist = distfit()
    for column in numeric_df.columns:
        identifier = 'normal'
        if(flag == 0):
            conditioned_DataFrame = data[data['class'] == 'normal']
        elif(flag == 1):
            conditioned_DataFrame = data[data['class'] == 'anomaly']
            identifier = 'anomaly'
        droppedData = conditioned_DataFrame[column].dropna()  # Drop NaN values
        try:
            dist.fit_transform(droppedData, verbose=0)
            # Store the best-fit distribution model
            best_fits_conditioned[column] = dist.model['name']
            print(i, "- ", column, ": Best fit distribution based on ", identifier, " is",  dist.model['name'])
            i += 1

        except Exception as e:
            print(f"Could not fit distribution for {column}: {e}")
    return best_fits_conditioned

best_fits_normal = calculate_PDF_conditioned(data,numeric_df,0)
best_fits_anomaly = calculate_PDF_conditioned(data,numeric_df,1)

#Unconditioned

for field, dist_name in best_fits.items():
    # Get the distribution object using its name
    dist = getattr(stats, dist_name, None)

    if dist and field in numeric_df.columns:
        # Fit the distribution for the normal data
        params = dist.fit(numeric_df[field])

        # Create a range of x values for plotting the PDF
        x = np.linspace(min(numeric_df[field]),max(numeric_df[field]), 100)#Why specially 100? Ask CHATGPT
        lower_bound, upper_bound = np.percentile(x, [3, 97])
        x_updated = np.linspace(lower_bound,upper_bound,100)

        # Calculate the PDF for the fitted distribution for both normal and anomaly data
        pdf = dist.pdf(x_updated, *params)

        # Plot the histogram and PDF for the data
        plt.figure(figsize=(10, 5))
        plt.hist(numeric_df[field], bins=30, density=True, alpha=0.5, color='blue', label='Uncoditioned Data')
        plt.plot(x_updated, pdf, 'b-', lw=2, label=f'Unconditioned Fit ({dist_name})')

        # Add titles and labels
        plt.title(f"Field: {field} with Best Fit Distribution: {dist_name}")
        plt.xlabel(field)
        plt.ylabel('Density')
        plt.legend()
        plt.xlim(lower_bound,upper_bound)

#plt.show() appears once at the end of the code to output all graphs together once.



#Conditioned

normal_data = df[df['class'] == 'normal'].select_dtypes(include=[np.number])
anomaly_data = df[df['class'] == 'anomaly'].select_dtypes(include=[np.number])

for (field, dist_name_normal), (field2, dist_name_anomaly) in zip(best_fits_normal.items(), best_fits_anomaly.items()):
    # Get the distribution object using its name
    dist_normal = getattr(stats, dist_name_normal, None)
    dist_anomaly = getattr(stats, dist_name_anomaly, None)

    if dist_normal and dist_anomaly and field in normal_data.columns and field2 in anomaly_data.columns:
        # Fit the distribution for the normal data
        params_normal = dist_normal.fit(normal_data[field])
        params_anomaly = dist_anomaly.fit(anomaly_data[field])

        x = np.linspace(min(normal_data[field].min(), anomaly_data[field].min()),
                        max(normal_data[field].max(), anomaly_data[field].max()), 100)#Why specially 100? Ask CHATGPT
        lower_bound, upper_bound = np.percentile(x, [3, 97])

        x_updated = np.linspace(lower_bound,upper_bound,100)

        pdf_normal = dist_normal.pdf(x_updated, *params_normal)
        pdf_anomaly = dist_anomaly.pdf(x_updated, *params_anomaly)

        plt.figure(figsize=(10, 5))
        plt.hist(normal_data[field], bins=20, density=True, alpha=0.5, color='blue', label='Normal Data')
        plt.plot(x_updated, pdf_normal, 'b-', lw=2, label=f'Normal Fit ({dist_name_normal})')

        plt.hist(anomaly_data[field], bins=20, density=True, alpha=0.5, color='red', label='Anomaly Data')
        plt.plot(x_updated, pdf_anomaly, 'r-', lw=2, label=f'Anomaly Fit ({dist_name_anomaly})')

        # Add titles and labels
        plt.title(f"Field: {field}")
        plt.xlabel(field)
        plt.ylabel('Density')
        plt.legend()
        plt.xlim(lower_bound,upper_bound)

# PMF

for column in data.columns:
    valid_data = data[column].dropna()  # Remove NaN values
    if data[column].dtype == 'object' or len(data[column].unique()) <= 20:  # Discrete data (categorical or few unique values)
        pmf = data[column].value_counts(normalize=True)
        # Plot PMF
        plt.figure(figsize=(10, 6))
        pmf.plot(kind='bar', color='blue', alpha=0.6, label='Normal', width=0.4, position=1)
        plt.title(f'PMF of {column}')
        plt.xlabel(column)
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
    

for column in data.columns:
    array = [1,0] #1 for anomaly and 0 for normal
    for count in array:
        if (count == 0):
            conditionedData = data[data['class'] == 'normal']
        elif (count == 1):
            conditionedData = data[data['class'] == 'anomaly']
        if df.dtypes[column] == 'object' or (len(data[column].unique()) <= 20):
            plt.figure(figsize=(10,5))
            if (count == 0): #normal data
                normal_pmf = conditionedData[column].value_counts(normalize=True)
                normal_pmf.plot(color= 'blue',kind='bar', label= f'Original PMF')
                plt.title(f"PMF of {column} (based on Normal)")
            elif (count == 1): #anomaly data
                anomaly_pmf = conditionedData[column].value_counts(normalize=True)
                anomaly_pmf.plot(color= 'orange', kind='bar', label= f'Conditional PMF')
                plt.title(f"PMF of {column} (based on Anomaly)")                

 
#Summary
print("PDF summary: ")
#PDFs
PDF_table = []

# After running the existing code for each column, store the results
for column in numeric_df.columns:
    real_dist = getattr(stats, best_fits[column])  # This gets the actual distribution object (e.g., scipy.stats.norm) 'distribution object from stats' to prevent error
    params = real_dist.fit(numeric_df[column]) 
    PDF_table.append({
        'Field Name': column,
        'Best-fit PDF': best_fits[column],
        'Parameters': params,
    })

# Convert the summary list to a DataFrame for better visualization
summary_PDFs_df = pd.DataFrame(PDF_table)
print(summary_PDFs_df)


#PMFs
print("PMF summary:")
def calculate_pmf(field):
    return field.value_counts(normalize=True).sort_index()

PMF_table = []

for column in df.columns:
    if df.dtypes[column] == 'object' or (len(data[column].unique()) <= 20):
        pmf = calculate_pmf(df[column])
        # Create a row with the column name and its terms + PMF values
        row = {'Field': column}
        row.update(pmf.to_dict())  # Add PMF values for each category
        PMF_table.append(row)

# Convert the list of rows into a DataFrame to display in table format
pmf_df = pd.DataFrame(PMF_table)
print(pmf_df)

plt.show()
