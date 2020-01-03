import numpy as np
import matplotlib.pyplot as plt

from method2d import *
import figure2d as F
from IoUtest import CalcIoU, CalcIoU2, LastIoU
from GA import *
from MakeDataset import MakePointSet


# 標識の点群作成
fig, sign, AABB = MakePointSet(0, 500)
# 目標点群プロット
X1, Y1= Disassemble2d(sign)
plt.plot(X1, Y1, marker=".",linestyle="None",color="yellow")

# 推定図形プロット
points2 = ContourPoints(fig.f_rep, AABB=AABB, grid_step=1000, epsilon=0.01, down_rate = 0.5)
X2, Y2= Disassemble2d(points2)
plt.plot(X2, Y2, marker="o",linestyle="None",color="red")

plt.show()

print(fig.p)
print("ans:{}".format(CalcIoU2(sign, fig, flag=True)))

a = input()

# GAにより最適パラメータ出力
#best = GA(sign)
best = EntireGA(sign)
print("="*100)
#fig = F.tri([  4.69031672, -31.1625267 ,   0.41589902,   0.38685724])
print(fig.p)
print("ans:{}".format(CalcIoU2(sign, fig, flag=True)))
fig2 = F.circle(best[0])
print(best)
print("opt:{}".format(CalcIoU2(sign, fig2, flag=True)))

print(LastIoU(fig, fig2, AABB))