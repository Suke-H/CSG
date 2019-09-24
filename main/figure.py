from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def sphere(theta, phi, r=1, r0=[0,0,0]):
    x = r * np.cos(theta) * np.cos(phi) + r0[0]
    y = r * np.cos(theta) * np.sin(phi) + r0[1]
    z = r * np.sin(theta) + r0[2]

    return x, y, z

def plane(X, Y, n, d):
    return (d - n[0]*X - n[1]*Y)/n[2]


"""
theta = np.arange(-np.pi, np.pi, 0.01)
phi = np.arange(-2*np.pi, 2*np.pi, 0.02)

#二次元メッシュにしないといけない
T, P = np.meshgrid(theta, phi)

r0 = [1,2,3]

X, Y, Z = sphere(T, P, 2, r0)

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.scatter(r0[0] , r0[1] , r0[2],  color='green')
ax.plot_wireframe(X, Y, Z)
plt.show()
"""