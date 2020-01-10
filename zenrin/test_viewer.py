import numpy as np
import matplotlib.pyplot as plt

# main
from method import *
from PreProcess import PreProcess
from ransac import RANSAC
from Projection import PlaneProjection

# zenrin

import figure2d as F
from IoUtest import CalcIoU, CalcIoU2, LastIoU
from GA import *
from MakeDataset import MakePointSet



def PlaneViewer(path, savepath):
    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)

    #軸にラベルを付けたいときは書く
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    #点群,法線, 取得
    points, normals = PreProcess(path)

    # 平面検出
    plane, index, num = RANSAC(points, normals)

    # フィット点を平面射影
    plane_points, UVvector = PlaneProjection(points[index], plane)

    # 点群描画
    X, Y, Z = Disassemble(points)
    MX, MY, MZ = X[index], Y[index], Z[index]
    ax.plot(X, Y, Z, marker="o", linestyle='None', color="white")
    ax.plot(MX, MY, MZ, marker=".", linestyle='None', color="red")

    PX, PY, PZ = Disassemble(plane_points)
    ax.plot(PX, PY, PZ, marker=".", linestyle='None', color="blue")

    # 最適化された図形を描画
    plot_implicit(ax, plane.f_rep, points, AABB_size=1, contourNum=15)

    plt.show()

    # 射影点群描画
    UX, UY = Disassemble2d(UVvector)
    plt.plot(UX, UY, marker="o",linestyle="None",color="red")

    plt.show()

    np.save(savepath, UVvector)

# PlaneViewer("data/FC_00652.jpg_0.txt", "data/FC_00652")
# PlaneViewer("data/点群データFC_00587.jpg_0.txt", "data/FC_00587")

fig_type = 1

#標識の点群作成
#fig, sign, AABB = MakePointSet(fig_type, 500)

#sign = np.load("data/FC_00587.npy")
#print(sign.shape)

#print(CalcIoU2(sign, fig, True))

# 目標と、枠
fig = F.tri([0,0,1,0])
out_fig = F.tri([0,0,1.2,0])

sign = InteriorPoints(fig.f_rep, [-2, 2, -2, 2], 500)
out_shape = ContourPoints(out_fig.f_rep)

print(CalcIoU3(sign, out_fig, fig,  True))


# 点群プロット
X1, Y1= Disassemble2d(sign)
plt.plot(X1, Y1, marker="o",linestyle="None",color="orange")
X2, Y2= Disassemble2d(out_shape)
plt.plot(X2, Y2, marker="o",linestyle="None",color="red")



# figure = F.rect([0.5434855012414951, 0.0631636962438673, 1.7262828639445207, 3.484746973450081, 2.730669636231897])

#plot_implicit2d(figure.f_rep, sign ,AABB_size=2)

plt.show()


# GAにより最適パラメータ出力
#best = GA(sign)
best = EntireGA(sign)
print("="*100)

print(best[fig_type])

best_circle = F.circle(best[0])
best_tri = F.tri(best[1])
best_rect = F.rect(best[2])

AABB = [-2,2,-2,2]

print(LastIoU(fig, best_circle, AABB))
print(LastIoU(fig, best_tri, AABB))
print(LastIoU(fig, best_rect, AABB))