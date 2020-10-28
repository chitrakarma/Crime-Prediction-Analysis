from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

np.random.seed(0)

data_set = pd.read_csv('data set/Datasets for prediction/random_forest.csv')

# print(data_set.head())
# print(iris.feature_names)
# print(df.head())
# df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

data_set['is_train'] = np.random.uniform(0, 1, len(data_set)) <= .75

# print(data_set.head())

# CRIME TYPE
data_set['State'] = pd.factorize(data_set['State'])[0]
data_set['Region'] = pd.factorize(data_set['Region'])[0]
print(data_set.head())

X_train, X_test = data_set[data_set['is_train'] == True], data_set[data_set['is_train'] == False]

print('No. of obs. in the training data: ', len(X_train))
print('No. of obs. in the testing data: ', len(X_test))

features = data_set.columns[1:8]
print(features)

y_train = X_train['Type']

y_test = X_test['Type']
X_train = X_train[features]
X_test = X_test[features]

print(len(X_train))
print(y_train)
print(len(X_test))
print(y_test)

clf = RandomForestClassifier(n_jobs=5, random_state=0)

clf.fit(X_train, y_train)

print(clf.predict(X_test))

print(clf.predict_proba(X_test[features])[0:10])

preds = clf.predict(X_test[features])
# print(preds)
# preds = iris.target_names[clf.predict(X_test[features])]
#
# # Find accuracy which is 93%
print(pd.crosstab(y_test, clf.predict(X_test), rownames=['Actual Species'], colnames=['Predicted Species']))

print(accuracy_score(y_test, preds))

# REGION WISE

# print(list(data_set['State']))
# data_set['State'] = pd.factorize(data_set['State'])[0]
# print(data_set.head())
#
# X_train, X_test = data_set[data_set['is_train'] == True], data_set[data_set['is_train'] == False]
#
# print('No. of obs. in the training data: \n', len(X_train))
# print('No. of obs. in the testing data: \n', len(X_test))
#
# features = data_set.columns[:7]
# print(features)
# y_train = pd.factorize(X_train['Region'])[0]
#
# y_test = pd.factorize(X_test['Region'])[0]
# X_train = X_train[features]
# X_test = X_test[features]
#
# print(len(X_train))
# print(len(y_train))
# print(len(X_test))
# print(y_test)
#
# clf = RandomForestClassifier(n_jobs=2, random_state=0)
#
# clf.fit(X_train, y_train)
#
# print(clf.predict(X_test))
#
# print(clf.predict_proba(X_test[features])[0:10])
#
# preds = clf.predict(X_test[features])
# # print(preds)
# # preds = iris.target_names[clf.predict(X_test[features])]
# #
# # # Find accuracy which is 93%
# print(pd.crosstab(y_test, clf.predict(X_test), rownames=['Actual Species'], colnames=['Predicted Species']))
#
# print(accuracy_score(y_test, preds))