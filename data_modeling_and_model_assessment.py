import pandas as pd
# Display all the columns
pd.set_option("display.max_columns", None)
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics, linear_model
from imblearn.over_sampling import RandomOverSampler, SMOTE
from matplotlib import pyplot as plt

# Function for ordinal encoding of a df
def ordinal_encoding(df):
    enc = OrdinalEncoder()
    columns = list(df_learn.columns)
    for column in columns:
        column_numpy = df[column].to_numpy()
        column_numpy = column_numpy.reshape(-1, 1)
        df[column] = enc.fit_transform(column_numpy)
    return df
    
def metrics_to_list(df_target, df_data):
    accuracy = round(metrics.accuracy_score(df_target, clf.predict(df_data)), 2)
    precision = round(metrics.precision_score(df_target, clf.predict(df_data)), 2)
    f1_score = round(metrics.f1_score(df_target, clf.predict(df_data)), 2)
    metrics_list = [accuracy, precision, f1_score]
    return metrics_list
  
# Ordinal encoding for df_learn
df_learn = ordinal_encoding(df_learn)      
        
# Prepare a dataframe with all the features
df_learn_data = df_learn.drop(columns = ["class"])

# Prepare a dataframe with the class
df_learn_target = df_learn["class"]

# Ordinal encoding for df_test
df_test = ordinal_encoding(df_test) 

# Prepare a dataframe with all the features
df_test_data = df_test.drop(columns = ["class"])

# Prepare a dataframe with the class
df_test_target = df_test["class"]

# A random forest could be interesting with so many features
clf = RandomForestClassifier(max_depth = 100)
clf.fit(df_learn_data, df_learn_target)

# Store the metrics
rf_metrics = metrics_to_list(df_test_target, df_test_data)

# Adaboost could be interesting as it decreases the prediction error of the minority class
clf = AdaBoostClassifier(n_estimators = 100)
clf.fit(df_learn_data, df_learn_target)

# Store the metrics
ada_metrics = metrics_to_list(df_test_target, df_test_data)

# Build a linear regression to better understand which features are important
regr = linear_model.LinearRegression()
regr.fit(df_learn_data, df_learn_target)

# Show which features are more important
importance = regr.coef_
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Use the SMOTE method to oversample the minority class, even if the f1 scores were better with the 0.5, 0.4 helps 
# avoiding overfitting
oversample = SMOTE(sampling_strategy = 0.4)

# Resample both the datasets derived from df_learn and df_test
df_learn_data, df_learn_target = oversample.fit_resample(df_learn_data, df_learn_target)
df_test_data, df_test_target = oversample.fit_resample(df_test_data, df_test_target)

# A random forest could be interesting with so many features
clf = RandomForestClassifier(max_depth = 100)
clf.fit(df_learn_data, df_learn_target)

# Store the metrics
rf_smote_metrics = metrics_to_list(df_test_target, df_test_data)

# Adaboost could be interesting as it decreases the prediction error of the minority class
clf = AdaBoostClassifier(n_estimators = 100)
clf.fit(df_learn_data, df_learn_target)

# Store the metrics
ada_smote_metrics = metrics_to_list(df_test_target, df_test_data)

# Build a linear regression to better understand which features are important
regr = linear_model.LinearRegression()
regr.fit(df_learn_data, df_learn_target)

# Show which features are more important
importance = regr.coef_
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# List of columns to build a dataframe with all the models results
columns_df = ["RF", "RF SMOTE", "ADA Boost", "ADA Boost SMOTE"]

# Index of the dataframe
index_df = ["accuracy", "precision", "f1 score"]

# Create a dataframe with all the models results
df_scores = pd.DataFrame(list(zip(rf_metrics , rf_smote_metrics, ada_metrics, ada_smote_metrics)),columns = columns_df)
df_scores.index = index_df

print(df_scores)

# Retrieve the features
features = np.array(df_learn_data.columns)

# Create a dataframe composed of both the features and their coefficients in the linear regression
df_linear_regression = pd.DataFrame({"coefficients": importance, "features": features}, columns=["coefficients", "features"])

# 5 highest coefficients
df_linear_regression_first_5 = df_linear_regression.sort_values(by = "coefficients", ascending = False).reset_index(drop = True).head(5)

df_linear_regression_first_5["coefficients"] = df_linear_regression_first_5["coefficients"].apply(lambda x: round(x, 3))

print(df_linear_regression_first_5)

# 5 lowest coefficients
df_linear_regression_last_5 = df_linear_regression.sort_values(by = "coefficients", ascending = True).reset_index(drop = True).head(5)

df_linear_regression_last_5["coefficients"] = df_linear_regression_last_5["coefficients"].apply(lambda x: round(x, 3))

print(df_linear_regression_last_5)
