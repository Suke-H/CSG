import numpy as np
import matplotlib.pyplot as plt

from method2d import *
import figure2d as F
from IoUtest import CalcIoU, CalcIoU2
from GA import *

###GA######################################################
# # 標識

# #C1 = F.rect([0,0,1,1,np.pi/4])
# C1 = F.tri([0,0,1,np.pi/3])
# sign = InteriorPoints(C1.f_rep, bbox=(-2, 2) ,grid_step=100, epsilon=0.01, down_rate = 0.5)
# signX, signY = Disassemble2d(sign)
# plt.plot(signX, signY, marker=".",linestyle="None",color="yellow")
# plt.show()

# #test = F.tri([ 0.1266216 , -0.00942684,  0.97585683,  0.98332958])
# test = F.tri([-0.11409518,  0.06197383,  0.9390659 ,  1.08170844])

# #test = F.circle([-0.06751492266443182, 1.4380856028775904, 0.04503599669797543])

# #F1 = F.tri([-0.53995553,0.53059937,3.64442983,0.22395693])
# print("ans:{}".format(CalcIoU2(sign, C1)))
# print("ans:{}".format(CalcIoU2(sign, test)))

# #best = GA(sign)
# sankaku = EntireGA(sign, path="../data/結果/GA8/tri2.csv")
# print(sankaku)
# #print("ans:{}".format(CalcIoU(sign, C1)))

C1 = F.tri([0,0,1,np.pi/9])
sign = InteriorPoints(C1.f_rep, bbox=(-2, 2) ,grid_step=100, epsilon=0.01, down_rate = 0.5)
X, Y = Disassemble2d(sign)
plt.plot(X, Y, marker=".",linestyle="None",color="yellow")

OBBmax, OBBmin, OBB, l1, area = buildOBB(sign)
AABBmax, AABBmin, AABB, l2, area = buildAABB(sign)

points = InteriorPoints(OBB.f_rep, bbox=(-5, 5) ,grid_step=1000, epsilon=0.01, down_rate = 0.5)
X, Y = Disassemble2d(points)
plt.plot(X, Y, marker=".",linestyle="None",color="orange")

print(l1, l2)
OBBViewer(OBBmax, OBBmin)
AABBViewer(AABBmax, AABBmin)
plt.show()