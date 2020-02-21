import numpy as np
import matplotlib.pyplot as plt

# main
from method2d import *
from PreProcess import NormalEstimate
from ransac import RANSAC
from Projection import Plane2DProjection, Plane3DProjection
from MakeDataset import MakePointSet

# zenrin
import figure2d as F
from IoUtest import calc_score

# fig, points, AABB, trueIndex = MakePointSet(2, 500)
# print(fig.p)
# print(AABB)
# # X, Y = Disassemble2d(points)
# # plt.plot(X, Y, marker=".", linestyle="None", color="blue")
# # figpoints = ContourPoints(fig.f_rep, AABB=AABB, grid_step=1000, epsilon=0.01)
# # Xf, Yf = Disassemble2d(figpoints)
# # plt.plot(Xf, Yf, marker=".", linestyle="None", color="black")
# # plt.show()
# #figure = F.tri([0, 0, 1, 0])
#
# score = calc_score(points, fig, True, AABB)
# print(score)