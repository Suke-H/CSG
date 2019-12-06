import numpy as np
import matplotlib.pyplot as plt

from method2d import *
import figure2d as F
from IoUtest import CalcIoU
from GA import *

###GA######################################################
# 標識
C1 = F.circle([0,0,1])
sign = InteriorPoints(C1.f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)

# ランダム図形生成
figures = CreateRandomPopulation(sign, 3)

# plot
signX, signY = Disassemble2d(sign)
points1 = ContourPoints(figures[0].f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
X1, Y1= Disassemble2d(points1)
points2 = ContourPoints(figures[1].f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
X2, Y2= Disassemble2d(points2)
points3 = ContourPoints(figures[2].f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
X3, Y3= Disassemble2d(points3)

plt.plot(signX, signY, marker=".",linestyle="None",color="yellow")
plt.plot(X1, Y1, marker="o",linestyle="None",color="blue")
plt.plot(X2, Y2, marker="o",linestyle="None",color="red")
plt.plot(X3, Y3, marker="o",linestyle="None",color="green")

print("IoU1:{}".format(CalcIoU(sign, figures[0])))
print("IoU2:{}".format(CalcIoU(sign, figures[1])))
print("IoU3:{}".format(CalcIoU(sign, figures[2])))
plt.show()