import numpy as np
import figure2 as F
from method import *

def norm_sphere(x, y, z):
    return 1-np.sqrt(x**2+y**2+z**2)

S1 = F.sphere([0,0,0,1])
S2 = F.sphere([1,0,0,0.5])
S3 = F.sphere([-1,0,0,0.5])

kirby = F.OR(S1, F.OR(S2, S3))

AND_S = F.AND(S1, S2)

S4 = F.sphere([np.sin(np.pi/6), np.cos(np.pi/6), 0, 1])
S5 = F.sphere([-np.sin(np.pi/6), np.cos(np.pi/6), 0, 1])

AND_BEN = F.AND(S1, F.AND(S4, S5))

P_Z0 = F.plane([0,0,-1,0])
P_Z1 = F.plane([0,0,1,1])
P_X0 = F.plane([-1,0,0,0])
P_X1 = F.plane([1,0,0,1])
P_Y0 = F.plane([0,-1,0,0])
P_Y1 = F.plane([0,1,0,1])

HALF_S = F.AND(S1, P_Z0)

def sh(x, y, z):
        return 37.29042182 - np.sqrt((x+3.10735045)**2 + (y-1.81359686)**2 + (z+110.75950196)**2)

def p_z0(x, y, z):
    return z
def p_z1(x, y, z):
    return 1-z
def p_x0(x, y, z):
    return x
def p_x1(x, y, z):
    return 1-x
def p_y0(x, y, z):
    return y
def p_y1(x, y, z):
    return 1-y

# f = d - (ax+by+cz)



CUBE = F.AND(F.AND(F.AND(F.AND(F.AND(P_Z0, P_Z1), P_X0), P_X1), P_Y0), P_Y1)

def sample_plane(x, y, z):
    return x + 4*y + 5*z + 6

SAMPLE_PLANE = F.plane([-1, -4, -5, 6])

"""
def AND(f1, f2):
    return lambda x,y,z: f1(x,y,z) + f2(x,y,z) - np.sqrt(f1(x,y,z)**2 + f2(x,y,z)**2)
"""

AND = lambda f1, f2: lambda x,y,z: f1(x,y,z) + f2(x,y,z) - np.sqrt(f1(x,y,z)**2 + f2(x,y,z)**2)

cube = AND(AND(AND(AND(AND(p_z0, p_z1), p_x0), p_x1), p_y0), p_y1)

cy = F.cylinder([0,0,0,0,0,1,1]).f_rep
cylin = AND(cy, p_z0)
cylinder = AND(cylin, p_z1)

co = F.cone([0,0,1.5,0,0,-1, np.pi/12]).f_rep
con = AND(co, p_z0)

