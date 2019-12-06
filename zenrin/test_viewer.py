import numpy as np
import matplotlib.pyplot as plt

from method2d import *
import figure2d as F
from IoUtest import CalcIoU
from GA import *

###GA######################################################
# 標識
C1 = F.tri([0,0,1,np.pi/3])
sign = InteriorPoints(C1.f_rep, grid_step=1000, epsilon=0.01, down_rate = 0.5)

print("ans:{}".format(CalcIoU(sign, C1)))

best = GA(sign)
print(best.fig_type, best.figure.p)
print("ans:{}".format(CalcIoU(sign, C1)))
"""
# ランダム図形生成
people = CreateRandomPopulation(sign, 3)

print(people[0].fig_type, people[0].figure.p)
print(people[1].fig_type, people[1].figure.p)

Crossover(people[0], people[1])

print(people[0].figure.p)
print(people[1].figure.p)


# plot
signX, signY = Disassemble2d(sign)
points1 = ContourPoints(people[0].figure.f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
X1, Y1= Disassemble2d(points1)
points2 = ContourPoints(people[1].figure.f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
X2, Y2= Disassemble2d(points2)
points3 = ContourPoints(people[2].figure.f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
X3, Y3= Disassemble2d(points3)

plt.plot(signX, signY, marker=".",linestyle="None",color="yellow")
plt.plot(X1, Y1, marker="o",linestyle="None",color="blue")
plt.plot(X2, Y2, marker="o",linestyle="None",color="red")
plt.plot(X3, Y3, marker="o",linestyle="None",color="green")

print("IoU1:{}".format(CalcIoU(sign, people[0].figure, flag=True)))
print("IoU2:{}".format(CalcIoU(sign, people[1].figure, flag=True)))
print("IoU3:{}".format(CalcIoU(sign, people[2].figure, flag=True)))
plt.show()

sorted_people, scores = Rank(people, sign)
print(scores)
"""