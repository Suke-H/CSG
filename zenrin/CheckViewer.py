import numpy as np

from method import *
from method2d import *
from Projection import Plane3DProjection

def CheckView(i, dir_path, out_path, fig_type, opti_fig_type):
    center_list = np.load(dir_path+"center.npy")
    para2d_list = np.load(dir_path+"para2d.npy")
    points3d_list = np.load(dir_path+"points3d.npy")
    AABB2d_list = np.load(dir_path+"AABB2d.npy")
    trueIndex_list = np.load(dir_path+"trueIndex.npy")
    plane_list = np.load(dir_path+"plane.npy")
    u_list = np.load(dir_path+"u.npy")
    v_list = np.load(dir_path+"v.npy")
    O_list = np.load(dir_path+"O.npy")

    opti_para2d_list = np.load(out_path+"opti_para2d.npy")
    opti_AABB2d_list = np.load(out_path+"opti_AABB2d.npy")
    opti_plane_list = np.load(out_path+"opti_plane.npy")
    opti_u_list = np.load(out_path+"opti_u.npy")
    opti_v_list = np.load(out_path+"opti_v.npy")
    opti_O_list = np.load(out_path+"opti_O.npy")

    ViewTest(points3d_list[i], trueIndex_list[i],  # 点群
            para2d_list[i], fig_type, plane_list[i], u_list[i], v_list[i], O_list[i], AABB2d_list[i], # 正解図形
            opti_para2d_list[i], opti_fig_type, opti_plane_list[i], opti_u_list[i], opti_v_list[i], opti_O_list[i], opti_AABB2d_list[i])# 検出図形


def ViewTest(points3d, trueIndex,  # 点群
            para2d, fig_type, plane_para, u, v, O, AABB2d, # 正解図形
            opti_para2d, opti_fig_type, opti_plane_para, opti_u, opti_v, opti_O, opti_AABB2d): # 検出図形

    
    # プロット
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    X, Y, Z = Disassemble(points3d[trueIndex])
    NX, NY, NZ = Disassemble(points3d[trueIndex==False])
    ax.plot(X, Y, Z, marker=".", linestyle='None', color="orange")
    ax.plot(NX, NY, NZ, marker=".", linestyle='None', color="blue")

    # 図形境界線作成
    if fig_type == 0:
        fig = F.circle(para2d)
    elif fig_type == 1:
        fig = F.tri(para2d)
    else:
        fig = F.rect(para2d)

    if opti_fig_type == 0:
        opti_fig = F.circle(opti_para2d)
    elif opti_fig_type == 1:
        opti_fig = F.tri(opti_para2d)
    else:
        opti_fig = F.rect(opti_para2d)

    goal2d = ContourPoints(fig.f_rep, AABB2d, grid_step=1000)
    center, goal3d = Plane3DProjection(goal2d, para2d, u, v, O)
    opti2d = ContourPoints(opti_fig.f_rep, opti_AABB2d, grid_step=1000)
    opti_center, opti3d = Plane3DProjection(opti2d, opti_para2d, opti_u, opti_v, opti_O)

    print(center, opti_center)

    GX, GY, GZ = Disassemble(goal3d)
    OX, OY, OZ = Disassemble(opti3d)

    ax.plot(GX, GY, GZ, marker=".", linestyle='None', color="red")
    ax.plot(OX, OY, OZ, marker=".", linestyle='None', color="green")
    ax.plot([center[0]], [center[1]], [center[2]], marker="o", linestyle='None', color="red")
    ax.plot([opti_center[0]], [opti_center[1]], [opti_center[2]], marker="o", linestyle='None', color="green")

    plt.show()
    plt.close()

CheckView(1, "data/dataset/3D/SET1/1/", "data/EntireTest/SET1/1/", 0, 0)