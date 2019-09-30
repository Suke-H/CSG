from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def MakePoints(fn, bbox=(-2.5,2.5), AABB_size=2):
    ''' 陰関数のグラフ描画
    fn  ...fn(x, y, z) = 0の左辺
    bbox ..x, y, zの範囲'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3

    #グラフの枠を作っていく
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    A_X = np.linspace(xmin, xmax, 100) # resolution of the contour
    A_Y = np.linspace(ymin, ymax, 100)
    A_Z = np.linspace(zmin, zmax, 100)
    B_X = np.linspace(xmin, xmax, 15) # number of slices
    B_Y = np.linspace(ymin, ymax, 15)
    B_Z = np.linspace(zmin, zmax, 15)
    #A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B_Z: # plot contours in the XY plane
        print(z)
        X,Y = np.meshgrid(A_X, A_Y)
        Z = fn(X,Y,z)
        np.savetxt("txt/"+str(z)+".txt", Z)
        index = np.where(Z==float(0))
        print(index)
        index = [(index[0][i], index[1][i]) for i in range(len(index[0]))]
        print(index)
        X1 = X[index]
        Y1 = Y[index]
        Z1 = np.array([z for i in range(len(index))])

        cset = ax1.contour(X, Y, Z+z, [z])
        cset = ax2.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B_Y: # plot contours in the XZ plane
        X,Z = np.meshgrid(A_X, A_Z)
        Y = fn(X,y,Z)
        cset = ax1.contour(X, Y+y, Z, [y], zdir='y')
        cset = ax3.contour(X, Y+y, Z, [y], zdir='y')

    for x in B_Z: # plot contours in the YZ plane
        Y,Z = np.meshgrid(A_Y, A_Z)
        X = fn(x,Y,Z)
        cset = ax1.contour(X+x, Y, Z, [x], zdir='x')
        cset = ax4.contour(X+x, Y, Z, [x], zdir='x')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.


    ax1.set_zlim3d(zmin,zmax)
    ax1.set_xlim3d(xmin,xmax)
    ax1.set_ylim3d(ymin,ymax)

    ax2.set_zlim3d(zmin,zmax)
    ax2.set_xlim3d(xmin,xmax)
    ax2.set_ylim3d(ymin,ymax)
    ax3.set_zlim3d(zmin,zmax)
    ax3.set_xlim3d(xmin,xmax)
    ax3.set_ylim3d(ymin,ymax)
    ax4.set_zlim3d(zmin,zmax)
    ax4.set_xlim3d(xmin,xmax)
    ax4.set_ylim3d(ymin,ymax)

    plt.show()

def norm_sphere(x, y, z):
    return 1-np.sqrt(x**2+y**2+z**2)

def sh(x, y, z):
        return x**2+y**2+z**2-1

MakePoints(norm_sphere)