# heart-disease
Heart Disease Prediction using Logistic Regression
This project aims to predict the presence of heart disease in individuals based on various health parameters using machine learning, specifically Logistic Regression.

Dataset
The dataset (data.csv) contains information about several health parameters and whether the individual has heart disease (target variable):

Features include attributes like age, sex, cholesterol levels, blood pressure, etc.
The target variable (target) indicates whether the person has heart disease (1) or not (0).
Workflow
Data Loading and Exploration:

Load the heart disease dataset from a CSV file into a Pandas DataFrame (heart_data).
Display dataset summary including the first and last few rows, dimensions, data types, and check for missing values.
Explore statistical measures of the data and check the distribution of the target variable (target).
Data Preprocessing:

Separate the dataset into features (X) and the target variable (Y).
Split the data into training and testing sets using train_test_split to evaluate model performance.
Model Training and Evaluation:

Initialize a Logistic Regression model (LogisticRegression) and train it using the training data (X_train, Y_train).
Evaluate the model's accuracy on both training and test sets using accuracy_score.
Prediction:

Demonstrate the model's predictive capabilities by providing an example input data.
Output the prediction result based on whether the individual is predicted to have heart disease or not.
Libraries Used
numpy and pandas for data manipulation and analysis.
sklearn for model selection (LogisticRegression), evaluation (train_test_split, accuracy_score), and preprocessing.
Conclusion
This project showcases the application of Logistic Regression for predicting heart disease based on individual health parameters. By training the model on historical data and evaluating its accuracy, it provides a reliable tool for assessing the risk of heart disease in individuals, thereby aiding in early detection and preventive healthcare measures.
