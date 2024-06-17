import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the CSV data into a Pandas DataFrame
heart_data = pd.read_csv('data.csv')

# Printing the first 5 rows of the dataset
print(heart_data.head())

# Printing the last 5 rows of the dataset
print(heart_data.tail())

# Displaying the number of rows and columns in the dataset
print(heart_data.shape)

# Getting information about the dataset
print(heart_data.info())

# Checking for missing values
print(heart_data.isnull().sum())

# Statistical measures about the data
print(heart_data.describe())

# Checking the distribution of the target variable
print(heart_data['target'].value_counts())
# 1 --> Defective Heart
# 0 --> Healthy Heart

# Separating the features (X) and target (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model initialization and training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on Training data : ', training_data_accuracy)

# Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on Test data : ', test_data_accuracy)

# Building a Predictive System
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')
