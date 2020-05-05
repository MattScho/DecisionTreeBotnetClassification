'''
Creates a Decision Tree to classify net flow data as beloning to  a botnet or benign usage
'''

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pickle as pkl
from sklearn.model_selection import train_test_split

# Read in files
FILE = ".\\data\\CTU-IoT-Malware-Capture-1-1\\bro\\conn.log.labeled"
files = [3,20,21,34,35]
headers = ["ts", "uid", "orig_ip","orig_port","resp_ip","resp_port","protocol","service","duration","orig_bytes","resp_bytes","conn_state","local_orig","local_resp","missed_bytes","history","orig_pkts","orig_ip_bytes","resp_pkts","resp_ip_bytes","tunnel_parents","label","detailed-label"]

# Format data to DataFrame
frame = pd.read_csv(FILE, skiprows=8, delimiter="\s+", names=headers)
frame.drop(frame.tail(1).index,inplace=True)
# Remove directly identifiable features
frame = frame.drop(["ts", "uid", "orig_ip", "resp_ip", "detailed-label"], axis=1)

# Inspect columns
for i in frame.columns:
    if frame[i].dtype =='object' and i != "detailed-label":
        frame[i] = frame[i].astype('category').cat.codes
		
# Build the data frame with the rest of the data
for file in files:
    print(file)
    frame1 = pd.read_csv(".\\data\\CTU-IoT-Malware-Capture-" + str(file) + "-1\\bro\\conn.log.labeled", skiprows=8, delimiter="\s+", names=headers)
    frame1.drop(frame1.tail(1).index,inplace=True)
    frame1 = frame1.drop(["ts", "uid", "orig_ip", "resp_ip", "detailed-label"], axis=1)

    for i in frame1.columns:
        if frame1[i].dtype =='object' and i != "detailed-label":
            frame1[i] = frame1[i].astype('category').cat.codes

    frame = frame.append(frame1)

# Split X and y
y = frame["label"].values[:-1]
X = frame.drop(["label",'service', 'orig_bytes', 'resp_bytes', 'local_orig', 'local_resp',
       'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
       'resp_ip_bytes', 'tunnel_parents'
], axis=1)
print(X.columns)

pd.set_option('display.max_columns', 500)


X=X.values[:-1]

len80 = int(len(y) * .8)

XTrain = X[:len80]
yTrain = y[:len80]
XTest = X[len80:len(X)]
yTest = y[len80:len(y)]


print(len(XTrain))
plt.figure()
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(XTest, yTest)
print(clf.feature_importances_)
print(accuracy_score(yTrain, clf.predict(XTrain)))

print(f1_score(yTrain, clf.predict(XTrain)))
print(confusion_matrix(yTrain, clf.predict(XTrain)))
pkl.dump(clf, open("DT.pkl", 'wb+'))