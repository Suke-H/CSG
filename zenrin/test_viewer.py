import numpy as np
import matplotlib.pyplot as plt

from method2d import *
import figure2d as F
from IoUtest import CalcIoU, CalcIoU2
from GA import *

###GA######################################################
# 標識

C1 = F.circle([1,1,2])
sign = InteriorPoints(C1.f_rep, bbox=(-2, 4) ,grid_step=1000, epsilon=0.01, down_rate = 0.5)

test = F.circle([-0.06751492266443182, 1.4380856028775904, 0.04503599669797543])

#F1 = F.tri([-0.53995553,0.53059937,3.64442983,0.22395693])
print("ans:{}".format(CalcIoU2(sign, C1)))
print("ans:{}".format(CalcIoU2(sign, test)))

#best = GA(sign)
sankaku = EntireGA(sign, path="../data/結果/GA7/test.csv")
#print(sankaku)
#print("ans:{}".format(CalcIoU(sign, C1)))



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
"""
a = 0.5
x1 = np.array([1,3])
x2 = np.array([2,4])
print(np.stack([x1,x2]))
xmax, xmin, _ = buildAABB(np.stack([x1,x2]))
#xmin = [1,3]
#xmax = [2,4]
print(xmin, xmax)

def BLX(x1, x2, xmin, xmax, a):
    r = Random(-a, 1+a)
    x = r*x1 + (1-r)*x2

    if any(xmin < x) and any(x < xmax):
        return x
        
    else:
        return BLX(x1, x2, xmin, xmax, a)

print(BLX(x1, x2, xmin, xmax, a))


parents = CreateRandomPopulation(4, [2, 2], [-2, -2], 2*np.sqrt(2), 0)

child = Crossover2(parents)
print(child)
"""