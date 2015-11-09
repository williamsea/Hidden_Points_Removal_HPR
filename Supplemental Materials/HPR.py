import numpy as np
import scipy as sp
import matplotlib
import math

class myPoint(object): 
	#Initializer
	def __init__(self, x, y, z):
		self.x=x
		self.y=y
		self.z=z

# Main
pointArray = []
for line in open("points.csv"):
	line = line.strip('\n')
	x,y,z = line.split(",")
	newPoint = myPoint(x,y,z)
	pointArray.append(newPoint)

dim = 3 # Dimension
numPts = len(pointArray) # Number of Points
C = myPoint(0,0,100) # View Point
param = math.pi # Parameter which indirectly sets the radius

Cpoints = np.tile(C, [numPts,1])
a = pointArray[0] - pointArray[1]

print a


