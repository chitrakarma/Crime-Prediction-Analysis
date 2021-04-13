import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn import tree

np.random.seed(0)

data_set = pd.read_csv('../Data Set/Datasets for prediction/random_forest.csv')

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

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

clf_tree = clf_entropy.fit(X_train, y_train)

y_pred = clf_entropy.predict(X_test)

# print(len(y_pred))

print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted'))
print(recall_score(y_test, y_pred, average='weighted'))
print(precision_score(y_test, y_pred, average='weighted'))
# print(confusion_matrix(y_test, y_pred))
tree.plot_tree(clf_tree)
# plt.savefig("sample.jpg")
plt.savefig("sample5.jpg", dpi=135)
# plt.savefig('1.svg', format='svg', dpi=1000)
# fig, ax = plt.subplots()
# fig.savefig('myimage.svg', format='svg', dpi=1200)
# plt.savefig('D:\Mayank\Repositories\Crime-Prediction-Analysis\Visuals', format='eps')
# plt.show()
