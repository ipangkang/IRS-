import scipy
from math import sin, cos, exp
from sympy import *

x = Symbol('x')
y = Symbol('y')

# 求导函数及导数
z1 = x ** 3
diff(z1, x)  # 关于x的导函数   即 3*x**2
diff(z1, x).subs(x, 5)  # 在x=5处的导数  即 75

# 求偏导函数及偏导数
z2 = x ** 2 + y ** 3
print(diff(z2, x))  # 关于x的偏导函数   即 2*x
print(diff(z2, x).subs(x, 4))  # 在x=4处的偏导数  即 8