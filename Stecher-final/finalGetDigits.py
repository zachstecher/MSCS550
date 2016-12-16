import numpy as np
from numpy import genfromtxt

def getDataSet ( ) :
    #  r e a d   d i g i t s   d a t a  &  s p l i t   i t   i n t o  X  and  y   f o r   t r a i n i n g   and   t e s t i n g
    dataset = genfromtxt ('features.csv', delimiter=' ')
    y = dataset [ : ,  0]
    X = dataset [ : ,  1 : ]
    
    dataset = genfromtxt ('features-t.csv', delimiter=' ')
    y_te = dataset[:, 0]
    X_te = dataset[:, 1:]
    return X, y, X_te, y_te
