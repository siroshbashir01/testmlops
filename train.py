from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
import os

import boto3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import awswrangler as wr
import os

s3_bucket = 's3://sagemaker-eu-west-1-978858584990/fraud-detection-model-testing-1-Inputdata'
df = f"{s3_bucket}"
print(df)
training_data = wr.s3.read_csv(path=df, sep=';')
print(training_data)
predicted_data_path = 's3://sagemaker-eu-west-1-978858584990/fraud-detection-batch-transformation/'
predicted_data = wr.s3.read_csv(path=df, sep=';')
print(predicted_data)
accuracy = accuracy_score(training_data['anomaly'], predicted_data['anomaly'])
print(accuracy)
precision = precision_score(
    predicted_data['anomaly'], training_data['anomaly'], average='macro')
samples = training_data['basetype'].count()
basetype_des = training_data['basetype'].describe()
basetype_des = basetype_des.astype(str)
print(basetype_des)

classification_report_al = classification_report(
    training_data['anomaly'], predicted_data['anomaly'])
with open(os.path.join('/Users/sirosh/Documents/cmldvc_dev_mlops', "{}.txt".format('matrices')), "w") as outfile:
    outfile.write("Training overall accuracy: %2.1f%%\n" % accuracy)
    outfile.write("precision: %2.1f%%\n" % precision)
    outfile.write("Total number of samples: %2.1f\n" % samples)
    outfile.write(str(classification_report_al))
    outfile.write(
        "-------------------Data description------------------ \n")
    outfile.write(str(basetype_des))

cm = confusion_matrix(training_data['anomaly'], predicted_data['anomaly'])
print(cm)
print(classification_report(
    training_data['anomaly'], predicted_data['anomaly']))
