'''
PythonTest: Play Around
'''

import numpy as np 
import math
import matplotlib.pyplot as plt

# Vector
row = np.array([1,2,3,4]) # Row Vector
col = np.array([1,2,3,4]).reshape(-1,1) # Col Vector
# print row, '\n', col

# Sequence
seq = range(1,11) # 1-10
jumpSeq = np.arange(1,11,3) # 1,4,7,10
# print seq, '\t', jumpSeq

# Concat
a = np.array([[1,2,3],[4,5,6]]) # Matrix Definition
b = np.array([[7,8,9],[10,11,12]])
concatDown = np.concatenate((a,b), axis=0) # in downward direction
concatRight = np.concatenate((a,b), axis=1) # in rightward direction
# print a, '\n', b, '\n', concatDown, '\n', concatRight

# Repeat
aRep = a.repeat(3) # output a single array
aRepDown = a.repeat(3, axis = 0)
aRepRight = a.repeat(3, axis = 1)
# print a, '\n', aRep, '\n', aRepDown, '\n', aRepRight

# Missing Elements
End = seq[-1]
LastTwo = seq[-2:]
ExceptFisrt = seq[1:]
# print seq, '\n', End, '\n', LastTwo, '\n', ExceptFisrt

# Vector Multiplication and Matrix Reshape/Transpose
mult = a*b # element wise calculation, equal to a.*b in Matlab
dot = np.dot(a, b.transpose()) # a * b, vector dot product, matrix calculation, (2,3)*(3,2)
# print a, '\n', b, '\n', b.reshape(3,2), '\n',  b.transpose(), '\n', mult, '\n', dot 
# note (b.reshape != b') here
# but b.transpose() == b'

# Matrix Binding and Flatten
vstack = np.vstack((a,b)) # need 2 sets of ()s, Stack vertically (Downwards) # equal to concatenate((a,b), axis=0) # [a ; b]
hstack = np.hstack((a,b)) # Stack horizontally (Rightwards) # equal to concatenate((a,b), axis=1) # [a , b]
dstack = np.dstack((a,b)) # Bind slices (three-way arrays)
stackInOne = np.concatenate((a,b), axis=None) # Concatenate matrices into one vector # [a(:), b(:)]
flatten = a.flatten()
# print a, '\n', b, '\n', vstack, '\n', hstack, '\n', dstack, '\n', stackInOne, '\n', flatten


# Array Creation
zeros = np.zeros((3,5), float)
ones = np.ones((3,5), float)
identity = np.identity(3)
diag = np.diag((3,4,5))
empty = np.empty((3,3)) # very small number but not zeros
# print zeros, '\n', ones, '\n', identity, '\n', diag, '\n', empty

# Indexing and Accessing Elements
matrix = np.array([[ 11, 12, 13, 14 ], [ 21, 22, 23, 24 ], [ 31, 32, 33, 34 ]])
# print matrix 
# print matrix[2,3] # index starts with 0
# print matrix[0] # first row
# print matrix[:,0] # first col
# print matrix[1:,] # except first row
# print matrix[-2:,:] # last two rows
# print matrix[::2, ] # every other row
# print matrix.take([0,2,3], axis = 1) # remove one col 


# Assignment
# print matrix
matrix[:,0] = 90
matrix[:,1] = [1,2,3]
# (matrix>10).choose(matrix, 10)
# matrix.clip(min = 10, max = 30)
# print matrix

# Sum
matrix = np.array([[ 11, 12, 13, 14 ], [ 21, 22, 23, 24 ], [ 31, 32, 33, 34 ]])
# print matrix
# print matrix.sum()
# print matrix.sum(axis = 0) # Sum in col direction
# print matrix.sum(axis = 1) # Sum in row direction
# print matrix.cumsum(axis = 0) # Cumulative sum (columns)

# Sort
a = np.array([[4,3,2],[2,8,6],[1,4,7]])


# Plots
x = np.arange(0,40,.5)
y = np.empty(shape = [0,len(x)]) # x.size
for a in x:
	y = np.append(y, math.sin(a/3) - math.cos(a/5))
print x
print y
plt.plot(x, y, 'o')
plt.grid()
plt.show()



