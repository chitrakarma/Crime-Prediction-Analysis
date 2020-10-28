from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

np.random.seed(0)

data_set = pd.read_csv('data set/test.csv')

data_set['is_train'] = np.random.uniform(0, 1, len(data_set)) <= .75

data_set['Region'] = pd.factorize(data_set['Region'])[0]
data_set['States/UTs'] = pd.factorize(data_set['States/UTs'])[0]
data_set['Type'] = pd.factorize(data_set['Type'])[0]

X_train, X_test = data_set[data_set['is_train'] == True], data_set[data_set['is_train'] == False]

features = data_set.columns[1:4]
print(features)

y_train = X_train['Region']

y_test = X_test['Region']
X_train = X_train[features]
X_test = X_test[features]

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

clf = RandomForestClassifier(n_jobs=3, random_state=0)

clf.fit(X_train, y_train)

# print(clf.predict(X_test))

print(clf.predict_proba(X_test)[0:10])

preds = clf.predict(X_test)
# print(preds)
# preds = iris.target_names[clf.predict(X_test[features])]
#
# # Find accuracy which is 93%
print(pd.crosstab(y_test, clf.predict(X_test), rownames=['Actual'], colnames=['Predicted']))

print(accuracy_score(y_test, preds))
print(f1_score(y_test, preds, average='weighted'))
print(recall_score(y_test, preds, average='weighted'))
print(precision_score(y_test, preds, average='weighted'))
print(confusion_matrix(y_test, preds))

# data_set = pd.read_csv('data set/test2014.csv')
#
# data_set['is_train'] = np.random.uniform(0, 1, len(data_set)) <= .75
#
# data_set['Region'] = pd.factorize(data_set['Region'])[0]
# data_set['States/UTs'] = pd.factorize(data_set['States/UTs'])[0]
# data_set['District'] = pd.factorize(data_set['District'])[0]
# data_set['Type'] = pd.factorize(data_set['Type'])[0]
# # print(data_set)
#
# X_train, X_test = data_set[data_set['is_train'] == True], data_set[data_set['is_train'] == False]
#
# # print('No. of obs. in the training data: ', len(X_train))
# # print('No. of obs. in the testing data: ', len(X_test))
#
# features = data_set.columns[2:5]
# # print(features)
#
# y_train = X_train['States/UTs']
#
# y_test = X_test['States/UTs']
# X_train = X_train[features]
# X_test = X_test[features]
#
# # print(len(X_train))
# # print(len(y_train))
# # print(len(X_test))
# # print(len(y_test))
#
# clf = RandomForestClassifier(n_jobs=3, random_state=0)
#
# clf.fit(X_train, y_train)
#
# # print(clf.predict(X_test))
#
# print(clf.predict_proba(X_test)[0:10])
#
# preds = clf.predict(X_test)
# # print(preds)
# # preds = iris.target_names[clf.predict(X_test[features])]
# #
# # # Find accuracy which is 93%
# print(pd.crosstab(y_test, clf.predict(X_test), rownames=['Actual'], colnames=['Predicted']))
#
# print(accuracy_score(y_test, preds))
# print(f1_score(y_test, preds, average='weighted'))
# print(recall_score(y_test, preds, average='weighted'))
# print(precision_score(y_test, preds, average='weighted'))
# print(confusion_matrix(y_test, preds))