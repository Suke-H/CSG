# sympyをインポート
import sympy

# 記号xを定義
sympy.var('x y z d epsilon alpha x0 y0 z0')


# f(x)を定義
dist = (d - x*x0 - y*y0 - z*z0) / epsilon
theta = sympy.acos(x*x0 + y*y0 + z*z0) / alpha
f = sympy.exp(-dist ** 2) + sympy.exp(-theta**2)

# f(x)を偏微分
df_dx = sympy.diff(f, x, 1)
df_dy = sympy.diff(f, y, 1)
df_dz = sympy.diff(f, z, 1)
df_dd = sympy.diff(f, d, 1)


print("df_dx = {}".format(df_dx))
print("df_dy = {}".format(df_dy))
print("df_dz = {}".format(df_dz))
print("df_dd = {}".format(df_dd))