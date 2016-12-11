import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

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
random_forest = RandomForestClassifier(n_estimators=100, max_features= 10, max_depth=50, min_samples_leaf=2)
random_forest.fit(x_tr, y_tr)
importances = random_forest.feature_importances_
#make permission 
y_pred = random_forest.predict_proba(x_test)
#submission
submission = pd.DataFrame(y_pred, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')
indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in random_forest.estimators_],
             axis=0)
# Print the feature ranking
print("Feature ranking:")
#x_tr.shape[1]
for f in range(10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_tr.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_tr.shape[1]), indices)
plt.xlim([-1, x_tr.shape[1]])
plt.show()