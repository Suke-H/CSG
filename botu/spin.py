import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from method import *

a = np.array([[1,0,0],[0,0,1],[0,1,0]])
P1 = np.array([[1,              0,               0],\
              [0,np.cos(np.pi/4),np.sin(np.pi/4)],\
              [0,-np.sin(np.pi/4),np.cos(np.pi/4)]])

P2 = np.array([[np.cos(np.pi/4),              0,-np.sin(np.pi/4)],\
              [               0,              1,               0],\
              [-np.sin(np.pi/4),              0, np.cos(np.pi/4)]])

P = np.dot(P1, P2)
print(P)

#print(np.linalg.inv(P))

new_a0 = np.dot(P2, a.T)
new_a1 = np.dot(P1, new_a0)
new_a = np.dot(P, a.T)
print(new_a)

#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

X0, Y0, Z0 = Disassemble(new_a0)
X1, Y1, Z1 = Disassemble(new_a1)
X, Y, Z = Disassemble(new_a)
O = [0,0,0]

ax.quiver(O, O, O, X0, Y0, Z0,  length=1,color='red', normalize=True)
ax.quiver(O, O, O, X1, Y1, Z1, length=1,color='green', normalize=True)
ax.quiver(O, O, O, X, Y, Z, length=1,color='blue', normalize=True)

plt.show()