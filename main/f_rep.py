from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def plot_implicit(fn, bbox=(-2.5,2.5)):
    ''' 陰関数のグラフ描画
    fn  ...fn(x, y, z) = 0の左辺
    bbox ..x, y, zの範囲'''
    #xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    max_p = np.array([ 36.948601 , 39.3424  , -75.815994])*4
    min_p = np.array([ -42.1959   , -37.601898 ,  -144.2155  ])*4

    print(max_p, min_p)

    xmax, ymax, zmax = max_p[0], max_p[1], max_p[2]
    xmin, ymin, zmin = min_p[0], min_p[1], min_p[2]

    #グラフの枠を作っていく
    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax1.contour(X, Y, Z+z, [z], zdir='z')
        cset = ax2.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax1.contour(X, Y+y, Z, [y], zdir='y')
        cset = ax3.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
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
        return 37.29042182 - np.sqrt((x+3.10735045)**2 + (y-1.81359686)**2 + (z+110.75950196)**2)

plot_implicit(sh)