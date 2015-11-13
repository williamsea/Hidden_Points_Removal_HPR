'''
@author: Hai Tang (haitang@jhu.edu)

Coding Challenge from Xometry Interview
Nov 1, 2015

Reference: 
Katz, Sagi, Ayellet Tal, and Ronen Basri. "Direct Visibility of Point Sets." 2007. 
'''

import csv
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D

'''
Function used to Import csv Points
'''
def importPoints(fileName, dim):

	p = np.empty(shape = [0,dim]) # Initialize points p
	for line in open(fileName): #For reading lines from a file, loop over the file object. Memory efficient, fast, and simple:
		line = line.strip('\n') # Get rid of tailing \n
		line = line.strip('\r') # Get rid of tailing \r
		x,y,z = line.split(",") # In String Format
		p = np.append(p, [[float(x),float(y),float(z)]], axis = 0) 

	return p

'''
Function used to Perform Spherical Flip on the Original Point Cloud
'''
def sphericalFlip(points, center, param):

	n = len(points) # total n points
	points = points - np.repeat(center, n, axis = 0) # Move C to the origin
	normPoints = np.linalg.norm(points, axis = 1) # Normed points, sqrt(x^2 + y^2 + (z-100)^2)
	R = np.repeat(max(normPoints) * np.power(10.0, param), n, axis = 0) # Radius of Sphere
	
	flippedPointsTemp = 2*np.multiply(np.repeat((R - normPoints).reshape(n,1), len(points[0]), axis = 1), points) 
	flippedPoints = np.divide(flippedPointsTemp, np.repeat(normPoints.reshape(n,1), len(points[0]), axis = 1)) # Apply Equation to get Flipped Points
	flippedPoints += points 

	return flippedPoints

'''
Function used to Obtain the Convex hull
'''
def convexHull(points):

	points = np.append(points, [[0,0,0]], axis = 0) # All points plus origin
	hull = ConvexHull(points) # Visibal points plus possible origin. Use its vertices property.

	return hull


'''
Main Function:
Apply Hidden Points Removal Operator to the Given Point Cloud
'''
def Main():
	myPoints = importPoints('points.csv', 3) # Import the Given Point Cloud

	C = np.array([[0,0,100]]) # View Point, which is well above the point cloud in z direction
	flippedPoints = sphericalFlip(myPoints, C, math.pi) # Reflect the point cloud about a sphere centered at C
	myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud


	# Plot
	fig = plt.figure(figsize = plt.figaspect(0.5))
	plt.title('Cloud Points With All Points (Left) vs. Visible Points Viewed from Well Above (Right)')
	
	# First subplot
	ax = fig.add_subplot(1,2,1, projection = '3d')
	ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	# Second subplot
	ax = fig.add_subplot(1,2,2, projection = '3d')
	for vertex in myHull.vertices[:-1]: # Exclude Origin, which is the last element
		ax.scatter(myPoints[vertex, 0], myPoints[vertex, 1], myPoints[vertex, 2], c='b', marker='o') # Plot visible points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	plt.show()

	return 

'''
Test Case:
A Sphere ranged from -1 to 1 in three axes with 961 Points is used for testing.
Viewed from well above: (0,0,10)
'''
def Test():
	myPoints = importPoints('sphere.csv', 3) # Import the Test Point Cloud

	C = np.array([[0,0,10]]) # 10 is well above the peak of circle which is 1
	flippedPoints = sphericalFlip(myPoints, C, math.pi)
	myHull = convexHull(flippedPoints)

	# Plot
	fig = plt.figure(figsize = plt.figaspect(0.5))
	plt.title('Test Case With A Sphere (Left) and Visible Sphere Viewed From Well Above (Right)')

	# First subplot
	ax = fig.add_subplot(1,2,1, projection = '3d')
	ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	# Second subplot
	ax = fig.add_subplot(1,2,2, projection = '3d')
	for vertex in myHull.vertices[:-1]:
		ax.scatter(myPoints[vertex, 0], myPoints[vertex, 1], myPoints[vertex, 2], c='b', marker='o') # Plot visible points
	ax.set_zlim3d(-1.5, 1.5)
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	plt.show()

	return 

'''
Onsite Coding Challenge
@author Hai Tang (haitang@jhu.edu)
Nov 12, 2015.
'''
def InvisiblePoints():
	myPoints = importPoints('points.csv', 3) # Import the Given Point Cloud

	############################ Method 1: Use a flag array indicating visibility, most efficent in speed and memory ############################
	flag = np.zeros(len(myPoints), int) # Initialize the points visible from possible 6 locations. 0 - Invisible; 1 - Visible.
	C = np.array([[[0,0,100]], [[0,0,-100]], [[0,100,0]], [[0,-100,0]], [[100,0,0]], [[-100,0,0]]])  # List of Centers
	for c in C:
		flippedPoints = sphericalFlip(myPoints, c, math.pi)
		myHull = convexHull(flippedPoints)
		visibleVertex = myHull.vertices[:-1] # indexes of visible points
		flag[visibleVertex] = 1
	invisibleId = np.where(flag == 0)[0] # indexes of the invisible points


	# Plot for method 1
	fig = plt.figure(figsize = plt.figaspect(0.5))
	plt.title('Cloud Points With All Points (Left) vs. Invisible Points (Right)')
	
	# First subplot
	ax = fig.add_subplot(1,2,1, projection = '3d')
	ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	# Second subplot
	ax = fig.add_subplot(1,2,2, projection = '3d')
	for i in invisibleId:
		ax.scatter(myPoints[i, 0], myPoints[i, 1], myPoints[i, 2], c='b', marker='o') # Plot visible points
	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	plt.show()

	############################ Method 2: Better with a loop, code simplified ############################

	# visibleAll = [] # Initialize the points visible from possible 6 locations
	# C = np.array([[[0,0,100]], [[0,0,-100]], [[0,100,0]], [[0,-100,0]], [[100,0,0]], [[-100,0,0]]]) 
	# for c in C:
	# 	flippedPoints = sphericalFlip(myPoints, c, math.pi)
	# 	myHull = convexHull(flippedPoints)
	# 	visibleVertex = myHull.vertices[:-1]
	# 	visibleAll = np.union1d(visibleAll, visibleVertex)

	#invisiblePoints = np.delete(myPoints, visibleAll, axis = 0)


	# # Plot for method 1 and method 2
	# fig = plt.figure(figsize = plt.figaspect(0.5))
	# plt.title('Cloud Points With All Points (Left) vs. Invisible Points (Right)')
	
	# # First subplot
	# ax = fig.add_subplot(1,2,1, projection = '3d')
	# ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
	# ax.set_xlabel('X Axis')
	# ax.set_ylabel('Y Axis')
	# ax.set_zlabel('Z Axis')

	# # Second subplot
	# ax = fig.add_subplot(1,2,2, projection = '3d')
	# ax.scatter(invisible[:, 0], invisible[:, 1], invisible[:, 2], c='b', marker='o') # Plot visible points
	# ax.set_xlabel('X Axis')
	# ax.set_ylabel('Y Axis')
	# ax.set_zlabel('Z Axis')

	# plt.show()

	############################ Method 3: Finished in 15min, working but not elegant ############################
	# C1 = np.array([[0,0,100]]) # View Point, which is well above the point cloud in z direction
	# C2 = np.array([[0,0,-100]])
	# C3 = np.array([[0,100,0]])
	# C4 = np.array([[0,-100,0]])
	# C5 = np.array([[100,0,0]])
	# C6 = np.array([[-100,0,0]])

	# flippedPoints = sphericalFlip(myPoints, C1, math.pi) # Reflect the point cloud about a sphere centered at C
	# myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud
	# visibleVertex1 = myHull.vertices[:-1]

	# flippedPoints = sphericalFlip(myPoints, C2, math.pi) # Reflect the point cloud about a sphere centered at C
	# myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud
	# visibleVertex2 = myHull.vertices[:-1]

	# flippedPoints = sphericalFlip(myPoints, C3, math.pi) # Reflect the point cloud about a sphere centered at C
	# myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud
	# visibleVertex3 = myHull.vertices[:-1]

	# flippedPoints = sphericalFlip(myPoints, C4, math.pi) # Reflect the point cloud about a sphere centered at C
	# myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud
	# visibleVertex4 = myHull.vertices[:-1]

	# flippedPoints = sphericalFlip(myPoints, C5, math.pi) # Reflect the point cloud about a sphere centered at C
	# myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud
	# visibleVertex5 = myHull.vertices[:-1]

	# flippedPoints = sphericalFlip(myPoints, C6, math.pi) # Reflect the point cloud about a sphere centered at C
	# myHull = convexHull(flippedPoints) # Take the convex hull of the center of the sphere and the deformed point cloud
	# visibleVertex6 = myHull.vertices[:-1]

	# temp1 = np.union1d(visibleVertex1, visibleVertex2)
	# temp2 = np.union1d(temp1, visibleVertex3)
	# temp3 = np.union1d(temp2, visibleVertex4)
	# temp4 = np.union1d(temp3, visibleVertex5)
	# visibleAll = np.union1d(temp4, visibleVertex6)

	#invisiblePoints = np.delete(myPoints, visibleAll, axis = 0)

	# # Plot for method 1 and method 2
	# fig = plt.figure(figsize = plt.figaspect(0.5))
	# plt.title('Cloud Points With All Points (Left) vs. Invisible Points (Right)')
	
	# # First subplot
	# ax = fig.add_subplot(1,2,1, projection = '3d')
	# ax.scatter(myPoints[:, 0], myPoints[:, 1], myPoints[:, 2], c='r', marker='^') # Plot all points
	# ax.set_xlabel('X Axis')
	# ax.set_ylabel('Y Axis')
	# ax.set_zlabel('Z Axis')

	# # Second subplot
	# ax = fig.add_subplot(1,2,2, projection = '3d')
	# ax.scatter(invisible[:, 0], invisible[:, 1], invisible[:, 2], c='b', marker='o') # Plot visible points
	# ax.set_xlabel('X Axis')
	# ax.set_ylabel('Y Axis')
	# ax.set_zlabel('Z Axis')

	# plt.show()

	return 

'''
Execution of Codes
'''
# Main()
# Test() 
InvisiblePoints()










