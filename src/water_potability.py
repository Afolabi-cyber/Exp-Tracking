import pandas as pd
import numpy as np
from dvclive import Live
import yaml

n_estimators = yaml.safe_load(open('C:/Users/user/Documents/exp-tracking-dvc/dvclive/params.yaml'))["n_estimators"]

data = pd.read_csv("C:/Users/user/Documents/exp-tracking-dvc/data/water_potability.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix, recall_score, f1_score

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].dtype == 'float64':
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df

  # fill missing values with median
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

x_train = train_processed_data.iloc[:, 0:-1].values
y_train = train_processed_data.iloc[:, -1].values

# n_estimators = 500

clf = RandomForestClassifier()
clf.fit(x_train, y_train)

#save
pickle.dump(clf, open('model.pkl', 'wb'))

x_test = test_processed_data.iloc[:, 0:-1].values
y_test = test_processed_data.iloc[:, -1].values

model = pickle.load(open('model.pkl', 'rb'))

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

with Live(save_dvc_exp=True) as live:
    live.log_metric("acc", accuracy)
    live.log_metric("Accuracy:", accuracy)
    live.log_metric("Recall:", recall)
    live.log_metric("Precision:", precision)
    live.log_metric("F1 Score:", f1_score)

    live.log_param('n_estimators', n_estimators)