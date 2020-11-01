from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model = GaussianNB()
np.random.seed(0)

data_set = pd.read_csv('data set/Datasets for prediction/random_forest.csv')

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

# print(X_train.shape[0] == y_train.shape[0])
#
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(accuracy_score(y_test, preds))
print(f1_score(y_test, preds, average='weighted'))
print(recall_score(y_test, preds, average='weighted'))
print(precision_score(y_test, preds, average='weighted'))
print(confusion_matrix(y_test, preds))
