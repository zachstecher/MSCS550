import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Any results you write to the current directory are saved as output.
# loading train data
traindata = pd.read_csv('train.csv')
x_tr = traindata.values[:, 2:]
y_tr = traindata.values[:, 1]
le = LabelEncoder().fit(traindata['species'])
scaler = StandardScaler().fit(x_tr)
x_tr = scaler.transform(x_tr)
#loading test data
testdata = pd.read_csv('test.csv')
x_test = testdata.drop(['id'], axis=1).values
x_test = scaler.transform(x_test)
test_ids = testdata.pop('id')
#Start learning
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_tr, y_tr)
#make permission 
y_pred = random_forest.predict_proba(x_test)
#submission
submission = pd.DataFrame(y_pred, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')