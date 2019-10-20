import numpy as np
import figure2 as F
from method import *

def norm_sphere(x, y, z):
    return 1-np.sqrt(x**2+y**2+z**2)

def sh(x, y, z):
        return 37.29042182 - np.sqrt((x+3.10735045)**2 + (y-1.81359686)**2 + (z+110.75950196)**2)

def p_z0(x, y, z):
    return z
def p_z1(x, y, z):
    return 1.5-z
def p_x0(x, y, z):
    return x
def p_x1(x, y, z):
    return 1.5-x
def p_y0(x, y, z):
    return y
def p_y1(x, y, z):
    return 1.5-y

def AND(f1, f2):
    return lambda x,y,z: f1(x,y,z) + f2(x,y,z) - np.sqrt(f1(x,y,z)**2 + f2(x,y,z)**2)

cube = AND(AND(AND(AND(AND(p_z0, p_z1), p_x0), p_x1), p_y0), p_y1)

cy = F.cylinder([0,0,0,0,0,1,1]).f_rep
cylin = AND(cy, p_z0)

co = F.cone([0,0,1.5,0,0,-1, np.pi/12]).f_rep
con = AND(co, p_z0)

