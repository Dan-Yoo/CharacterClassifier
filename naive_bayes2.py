import pandas as pd
import numpy as np
import csv as csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

dataset = "ds2"
base = "dataset/" + dataset + "/" + dataset
train_csv = base + "Train.csv"
val_csv = base + "Val.csv"
test_csv = base + "Test.csv"
val_result_csv = "predictions/ds2Val-nb.csv"
test_result_csv = "predictions/ds2Test-nb.csv"
model_name = "models/ds2-nb.joblib"

# import dataset
train = pd.read_csv(train_csv)
validation = pd.read_csv(val_csv)
test = pd.read_csv(test_csv)

X_train = train.iloc[:, :-1]
Y_train = train.iloc[:, -1]

X_val = validation.iloc[:, :-1]
Y_val = validation.iloc[:, -1]

X_test = test.iloc[:]

# create classifier
classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, Y_train)
# fit the classifier with the data
# train classfier
prediction_val = classifier.predict(X_val)
prediction_test = classifier.predict(X_test)

# write to val file
csv_file = open(val_result_csv, 'w')
csv_file.truncate()
# print
i = 1

with csv_file:
    writer = csv.writer(csv_file)

    while i < len(prediction_val):
        writer.writerow([i, prediction_val[i]])
        i += 1 
    
csv_file.close()

# write to test file
csv_file = open(test_result_csv, 'w')
csv_file.truncate()
# print
i = 1

with csv_file:
    writer = csv.writer(csv_file)

    while i < len(prediction_test):
        writer.writerow([i, prediction_test[i]])
        i += 1 
    
csv_file.close()

print(classifier.score(X_val, Y_val))

joblib.dump(classifier, model_name)