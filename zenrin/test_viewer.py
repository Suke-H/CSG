import numpy as np
import matplotlib.pyplot as plt

from method2d import *
import figure2d as F
from IoUtest import CalcIoU, CalcIoU2
from GA import *
from MakeDataset import MakePointSet


# # 標識の点群作成
# # 三角形作成
# C1 = F.rect([0,0,1,1,np.pi/4])
# # 内部点生成
# sign = InteriorPoints(C1.f_rep, bbox=(-2, 2) ,grid_step=100, epsilon=0.01, down_rate = 0.5)

# # GAにより最適パラメータ出力
# #best = GA(sign)
# best = EntireGA(sign)
# print(best)
# print("ans:{}".format(CalcIoU2(sign, C1)))

tri, sign, AABB = MakePointSet(500)

max_p = [AABB[1], AABB[3]]
min_p = [AABB[0], AABB[2]]

X, Y = Disassemble2d(sign)

plt.plot(X, Y, marker=".",linestyle="None",color="orange")
AABBViewer(max_p, min_p)

plt.show()

# C1 = F.rect([0, 0, 2, 1, np.pi/3])
# vert = C1.CalcVertices()
# print(vert)
# X, Y = Disassemble2d(vert)
# sign = MakePoints(C1.f_rep, AABB=[-1, 3, -1, 3] ,grid_step=1000, epsilon=0.01, down_rate = 0.5)
# signX, signY = Disassemble2d(sign)
# plt.plot(X, Y, marker="o",linestyle="None",color="red")
# plt.plot(signX, signY, marker=".",linestyle="None",color="orange")
# plt.show()