from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

points = np.random.rand(30,2)
hull = ConvexHull(points)

print points, hull.vertices

# plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1],'k-')

# plt.plot(points[hull.vertices,0], points[hull.vertices,1],'r--', lw=2)
# plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1],'ro')
# plt.show()


# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
