"""
这段代码用来对比按层横铺整个球面的效率
在按层横铺的算法中，一层圆圈能够组合出一个全覆盖的条带。
这个条带的长度为球体在这个纬度上的长度，宽度为重叠的宽度。
在圆圈的数量和半径一定时，在极限情况下，条带的宽度和长度的平方和是一定的。
因此，当圆圈平铺时的宽度为圆圈半径的根号2分之一时，理论效率最高。
"""
import numpy as np


def calculate_number_of_circles(radius: float, width:float) -> int:
    """
    给定圆圈半径（弧度制）和宽度（与圆圈半径的比值，小于1），计算铺满整个球需要多少个圆圈
    """
    result = 0
    for angle in np.arange(width*radius, np.pi, width*radius*2):
        r = np.sin(angle)
        c = 2*np.pi*r
        l = 2*radius*np.sqrt(1 - width**2)
        result += np.ceil(c/l)
    return result

for i in range(1, 10):
    for j in [1/2,  np.sqrt(2)/2, np.sqrt(3)/2]:
        print(calculate_number_of_circles(0.5*i*np.pi/180.0, j),end=" ")
    print()