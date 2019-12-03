import numpy as np

from ctypes import *
import io
import struct
from sys import getsizeof as S
"""
A = struct.pack('ffff', 1, 2, 3, 4)
print(A)
"""

fid = open("../data/tanaka.dat", 'rb')
for i in range(4):
    a = struct.unpack('<f', fid.read(4))
    print(a)
