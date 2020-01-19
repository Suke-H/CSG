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
from TransPix import MakeOuterFrame
from ClossNum import CheckClossNum, CheckClossNum2

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

fig_type = 0

#標識の点群作成
fig, sign, AABB = MakePointSet(fig_type, 500)

X1, Y1 = Disassemble2d(sign)
plt.plot(X1, Y1, marker="o",linestyle="None",color="orange")
plt.show()

out_points, out_area = MakeOuterFrame(sign, path="data/last/test.png")

print(fig.p)
print(AABB)
print(out_area)

np.save("data/last/sign", sign)
np.save("data/last/out", out_points)

# fig = F.tri([54.6477030005436, 61.480097301238054, 9.813207149437893, 1.0337986278688602])
# AABB = [43.71376348524868, 70.73072201781974, 50.70809854597516, 76.68202602668589]
# out_area = 154.63881276751636

# sign = np.load("data/last/sign.npy")
# out_points = np.load("data/last/out.npy")

print("="*50)

# 外枠内の点群だけにする
#inside = np.array([CheckClossNum(points[i], out_contour) for i in range(points.shape[0])])
inside = CheckClossNum2(sign, out_points)
sign = sign[inside]

print(CalcIoU3(sign, out_points, out_area, fig,  True))

# points = ContourPoints(fig.f_rep, AABB=AABB, grid_step=1000, epsilon=0.01)
# print(points.shape)

# X1, Y1 = Disassemble2d(points)
# X2, Y2 = Disassemble2d(out_points)
# plt.plot(X2, Y2, marker=".",linestyle="None",color="red")
# plt.plot(X1, Y1, marker="o",linestyle="None",color="orange")
# plt.show()


# GAにより最適パラメータ出力
#best = GA(sign)
best = EntireGA(sign, out_points, out_area)
print("="*100)

# print(best[0].figure.p)
# best_tri = F.tri(best[0].figure.p)
# print(LastIoU(fig, best_tri, AABB))

print(best[fig_type].figure.p)

best_circle = F.circle(best[0].figure.p)
best_tri = F.tri(best[1].figure.p)
best_rect = F.rect(best[2].figure.p)

print(LastIoU(fig, best_circle, AABB))
print(LastIoU(fig, best_tri, AABB))
print(LastIoU(fig, best_rect, AABB))