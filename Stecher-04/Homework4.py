import hw4genData
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from numpy import genfromtxt
from sklearn.model_selection import KFold
import time


# read digits data & split it into X (training input) and y (target output)
X, y, ytrue = hw4genData.genDataSet(1000)

X = X.reshape((len(X), 1))

bestk=[] #Holds every k-neighbors value, we will sort by score later to get the best 3 first
kc=0
bestklist=[] #Holds the best 3 k-neighbors from each repetition of the experiment for problem 2
for i in range(0, 100): #Performs the experiment 100 times for problem 2
  for n_neighbors in range(1,900,2):
    kf = KFold(n_splits=10)
    kscore=[]
    k=0
    for train, test in kf.split(X):
      #print("%s %s" % (train, test))
      X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    
      #time.sleep(100)
    
      # we create an instance of Neighbors Regressor and fit the data.
      clf = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
      clf.fit(X_train, y_train)
    
      kscore.append(clf.score(X_test,y_test))
      #print kscore[k]
      k=k+1
    
    #print (n_neighbors)
    bestk.append(sum(kscore)/len(kscore))
    #print bestk[kc]
    kc+=1

  idx = sorted(range(len(bestk)), key=bestk.__getitem__) #Sort the k-neighbors values
#  print(idx[-1]*2)
#  print(idx[-2]*2)
#  print(idx[-3]*2)

  #Add the best 3 values to our overall list for problem 2
  bestklist.append(idx[-1]*2)
  bestklist.append(idx[-2]*2)
  bestklist.append(idx[-3]*2)
  bestk[:] = [] #Clear bestk so we can run the experiment again on a clean array
  
print kscore[0] #Print the Eout
print bestklist #Print the list of best k-neighbors values from each experiment

#Plot a histogram of each 
plt.hist(bestklist)
plt.title("KNeighbor Results")
plt.xlabel("Number of Neighbors")
plt.ylabel("Occurrence")
plt.show()
  
