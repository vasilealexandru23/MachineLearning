import numpy as np
import matplotlib.pyplot as plt
import icp

# Define two sets of points A and B
# Create the initial array of data points
t = np.linspace(0, 2*np.pi, 10)
my_A = np.array([t, np.sin(t)]).T
A = np.column_stack(( t, np.sin(t) ))

# Define the rotation angle
theta = np.radians(30)
c, s = np.cos(theta), np.sin(theta)
rotation_matrix = np.array(((c, -s), (s, c)))

# Define translation vector
translation_vector = np.array([[2,0]])

# Apply the transformation and add randomness to get B
randomness = 0.3*np.random.rand(10,2)
B = np.dot(rotation_matrix, A.T).T + translation_vector + randomness

# Plot the original datasets
fig = plt.figure()
Aplot = plt.scatter( A[:,0], A[:,1], color = 'red')
Bplot = plt.scatter( B[:,0], B[:,1], color = 'blue')
plt.legend((Aplot,Bplot),("Point Set A","Point Set B"))
plt.title("Original Point Sets")
plt.show()

# Apply the ICP algorithm
R, t = icp.icp(A, B)

# Apply the transformation to A
A_transformed = np.dot(R, A.T).T + t

# Plot the transformed A and B
fig = plt.figure()
Aplot = plt.scatter( A_transformed[:,0], A_transformed[:,1], color = 'red')
Bplot = plt.scatter( B[:,0], B[:,1], color = 'blue')
plt.legend((Aplot,Bplot),("Point Set A","Point Set B"))
plt.title("Transformed Point Sets")
plt.show()
