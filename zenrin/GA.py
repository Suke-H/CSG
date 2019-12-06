import numpy as np

from method2d import *
import figure2d as F

def CreateRandomPerson(fig, max_p, min_p, l):
    # 円
    if fig==0:
        print("円")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < r < 2/3*l
        r = Random(0, 2/3*l)

        figure = F.circle([x,y,r])

    # 正三角形
    elif fig==1:
        print("正三角形")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < r < 2/3*l
        r = Random(0, 2/3*l)
        # 0 < t < pi/2
        t = Random(0, np.pi/2)

        figure = F.tri([x,y,r,t])

    # 長方形
    elif fig==2:
        print("長方形")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < w,h < l
        w = Random(0, 2/3*l)
        h = Random(0, 2/3*l)
        # 0 < t < pi/2
        t = Random(0, np.pi/2)

        figure = F.rect([x,y,w,h,t])

    return figure


def CreateRandomPopulation(points, num, fig=[0,1,2]):
    # AABB生成
    max_p, min_p, l = buildAABB(points)

    # ランダムに図形の種類を選択し、図形生成
    figure_list = [CreateRandomPerson(np.random.choice(fig), max_p, min_p, l) for i in range(num)]

    return figure_list
