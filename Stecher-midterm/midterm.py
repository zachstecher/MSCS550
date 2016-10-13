import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
from numpy import genfromtxt

class Perceptron:
    def __init__(self, N):
        # Random linearly separated data
        xA,yA,xB,yB = [random.uniform(-1, 1) for i in range(4)]
        self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        self.X = self.generate_points(N)
 
    def generate_points(self, N):
         #read digits data and split it into X (training input) and y (target output)
        X, y = self.import_data()
        bX = []
        for k in range(0,N) :
            bX.append((np.concatenate(([1], X[k,:])), y[k]))
            
        # this will calculate linear regression at this point
        X = np.concatenate((np.ones((N,1)), X),axis=1) # adds the 1 constant
        self.linRegW = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)   # lin reg
        return bX
 
    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(0,1)
        plt.ylim(-8,1)
        V = self.V
        a, b = -V[1]/V[2], -V[0]/V[2]
        l = np.linspace(-9,9)
        plt.plot(l, a*l+b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'o')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
                plt.savefig('p_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight')
 
    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error
 
    def choose_miscl_point(self, vec):
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False):
      # Initialize the weights to zeros
      #w = np.zeros(3)
      w = self.linRegW
      X, N = self.X, len(self.X)
      it = 0

      #Initialize variable for pocket algorithm
      self.bestW=w
      self.plaError=[]
      self.pocketError=[]
     
      #pocket algorithm
      self.plaError.append(self.classification_error(w))
      self.pocketError.append(self.plaError[it])
      while self.plaError[it]!=0:
          it+=1
          x,s=self.choose_miscl_point(w)
          w+=s*x
          self.plaError.append(self.classification_error(w))
          if(self.pocketError[it-1] > self.plaError[it]):
            self.pocketError.append(self.plaError[it])
            self.bestW=w
            print self.pocketError[it]
            its = it
            print it
          else:
            self.pocketError.append(self.pocketError[it-1])
          if it == 1000:
            print it
            break
      if save:
         self.plot(vec=self.bestW)
         plt.title('N = %s, Iteration %s\n' \
                  % (str(N),str(its)))
         plt.savefig('p_N%s_it%s' % (str(N),str(its)), \
                  dpi=200, bbox_inches='tight')
      self.w = w
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)

    def import_data(self):
        #read digits data and split it into X (training input) and y (target output)
        dataset = genfromtxt('features.csv', delimiter=' ')
        y = dataset[:,0]
        X = dataset[:, 1:]

        y[y<>1] = -1   #rest of numbers are negative class
        y[y==1] = +1   #0 is the positive class

        #plots data
        c0 = plt.scatter(X[y==-1,0], X[y==-1,1], s=20, color='r', marker='x')
        c1 = plt.scatter(X[y==1,0], X[y==1,1], s=20, color='b', marker='o')

        #displays legend
        plt.legend((c0, c1), ('All Other Numbers -1', 'Number Zero +1'), loc='upper right', scatterpoints=1, fontsize=11)

        #displays axis legends and title
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'Intensity and Symmetry of Digits')

        #saves he figure into a .pdf file
        plt.savefig('midterm.plot.pdf', bbox_inches='tight')
        plt.show()
        
        return X, y

def main():
    p = Perceptron(7291)
    p.pla(save=True)

main()
