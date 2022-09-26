from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
import sys

tf.reset_default_graph()

file_train = "JUN.txt"
file_test = "ACP1.txt"

dat1 = np.loadtxt(file_train, dtype=float)
dat2 = np.loadtxt(file_test, dtype=float)
#dat = np.loadtxt(file, dtype=float, delimiter=",", skiprows=0)
y_train = np.array(dat1[:,-1], dtype=int)
expression_train = np.array(dat1[:,:-1])

y_test = np.array(dat2[:,-1], dtype=int)
expression_test = np.array(dat2[:,:-1])

scaler = preprocessing.StandardScaler()
scaler.fit(expression_train)
expression_train = scaler.transform(expression_train)

scaler.fit(expression_test)
expression_test = scaler.transform(expression_test)

model = LogisticRegression(penalty='l1') 
model.fit(expression_train, y_train)
y_s=model.predict(expression_test)
   
acc = metrics.accuracy_score(y_test, y_s > 0.5)
auc = metrics.roc_auc_score(y_test, y_s)
prob_predict_y_validation = model.predict_proba(expression_test)
   #print("Case proportion in testing data:", round(sum(y_test[:,1])/np.shape(y_test)[0], 3)) 
predictions_validation = prob_predict_y_validation[:, 1]
with open('LR_JUN_ACP.txt', mode='a') as filename:
    for eee in range(len(y_test)):
        filename.write(str(y_test[eee])+' '+str(predictions_validation[eee])+'\n')