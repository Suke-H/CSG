import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

from loadOBJ import loadOBJ
import figure2 as F
from PreProcess2 import PreProcess2
from method import *
from figure_sample import *

points, X, Y, Z, normals, length = PreProcess2()
ax = ViewerInit(points, X, Y, Z)
#
figure = F.SPIN(S2, np.pi/4, 0, 0)
#figure = CYLINDER
#
plot_normal(ax, figure, X, Y, Z)
#plot_implicit(ax, figure.f_rep, bbox=(-1,1))
plt.show()