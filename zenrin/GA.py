import numpy as np

from method2d import *
import figure2d as F
from IoUtest import CalcIoU

class person:
    def __init__(self, fig_type, figure):
        self.fig_type = fig_type
        self.figure = figure
        self.score = 0
        self.scoreFlag = False

def CreateRandomPerson(fig_type, max_p, min_p, l):
    # 円
    if fig_type==0:
        #print("円")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < r < 2/3*l
        r = Random(0, 2/3*l)

        figure = F.circle([x,y,r])

    # 正三角形
    elif fig_type==1:
        #print("正三角形")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < r < 2/3*l
        r = Random(0, 2/3*l)
        # 0 < t < pi*2/3
        t = Random(0, np.pi*2/3)

        figure = F.tri([x,y,r,t])

    # 長方形
    elif fig_type==2:
        #print("長方形")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < w,h < l
        w = Random(0, l)
        h = Random(0, l)
        # 0 < t < pi
        t = Random(0, np.pi)

        figure = F.rect([x,y,w,h,t])

    return person(fig_type, figure)

def CreateRandomPopulation(points, num, fig=[0,1,2]):
    # AABB生成
    max_p, min_p, l = buildAABB(points)

    # ランダムに図形の種類を選択し、遺伝子たちを生成
    population = [CreateRandomPerson(np.random.choice(fig), max_p, min_p, l) for i in range(num)]

    return population

def Score(person, points):
    # scoreFlagが立ってなかったらIoUを計算
    if person.scoreFlag == False:
        person.score = CalcIoU(points, person.figure)
        person.scoreFlag = True

    return person.score

def Rank(people, points):
    # リストにスコアを記録していく
    score_list = [Score(people[i], points) for i in range(len(people))]
    # Scoreの大きい順からインデックスを読み上げ、リストに記録
    index_list = sorted(range(len(score_list)), reverse=True, key=lambda k: score_list[k])
    # index_listの順にPeopleを並べる
    return np.array(people)[index_list], np.array(score_list)[index_list]

# 図形パラメータのどれかを変更(今のところ図形の種類は変えない)
def Mutate(person, points, rate=1.0):
    # rateの確率で突然変異
    if np.random.rand() <= rate:
        # AABB生成
        max_p, min_p, l = buildAABB(points)
        # 図形パラメータの番号を選択
        index = np.random.choice([i for i in range(len(person.figure.p))])
        # 図形の種類にそって、選択したパラメータをランダムに変更
        print(index)

        # 円
        if person.fig_type == 0:
            # x
            if index == 0:
                person.figure.p[index] = Random(min_p[0], max_p[0])
            # y
            elif index == 1:
                person.figure.p[index] = Random(min_p[1], max_p[1])
            # r
            else:
                person.figure.p[index] = Random(0, 2/3*l)

        # 正三角形
        elif person.fig_type == 1:
            # x
            if index == 0:
                person.figure.p[index] = Random(min_p[0], max_p[0])
            # y
            elif index == 1:
                person.figure.p[index] = Random(min_p[1], max_p[1])
            # r
            elif index == 2:
                person.figure.p[index] = Random(0, 2/3*l)
            # t
            else:
                person.figure.p[index] = Random(0, np.pi*2/3)

        # 長方形
        elif person.fig_type == 2:
            # x
            if index == 0:
                person.figure.p[index] = Random(min_p[0], max_p[0])
            # y
            elif index == 1:
                person.figure.p[index] = Random(min_p[1], max_p[1])
            # w
            elif index == 2:
                person.figure.p[index] = Random(0, l)
            # h
            elif index == 3:
                person.figure.p[index] = Random(0, l)
            # t
            else:
                person.figure.p[index] = Random(0, np.pi/2)

# 同じ図形同士なら、場所を選択して交叉
# [x1,y1,r1][x2,y2,r2] -> 1を選択 -> [x1,y2,r2][x2,y1,r1]

# 違う図形同士なら、共通するパラメータを1つだけ交換
# [x1,y1,r][x2,y2,w,h,t] -> [x,y]が共通 -> 1を選択 -> [x1,y2,r][x2,y1,w,h,t]
# [x1,y1,r,t1][x2,y2,w,h,t2] -> [x,y,t]が共通 -> 2を選択 -> [x1,y1,r,t2][x2,y2,w,h,t1]
def Crossover(person1, person2, rate=1.0):
    # rateの確率で突然変異
    if np.random.rand() <= rate:

        f1, f2, p1, p2 = person1.fig_type, person2.fig_type, person1.figure.p, person2.figure.p

        # 同じ図形なら
        if f1 == f2:
            # 図形パラメータの番号を選択
            index = np.random.choice([i for i in range(len(p1))])
            # 同じ番号の場所を交代
            p1[index], p2[index] = p2[index], p1[index]

        # 違う図形なら
        else:
            # 円と正三角形 or 円と長方形
            if f1 == 0 or f2 == 0:
                # 図形パラメータの番号を選択
                index = np.random.choice([i for i in range(2)])
                # 同じ番号の場所を交代
                p1[index], p2[index] = p2[index], p1[index]

            # 正三角形と長方形
            else:
                # 図形パラメータの番号を選択
                index = np.random.choice([i for i in range(3)])

                # indexが0か1(=xかy)だったら無難に交換
                if index in [0,1]:
                    p1[index], p2[index] = p2[index], p1[index]

                # indexが2(=t)だったらまあ頑張って交換
                else:
                    if f1 == 1:
                        if p2[4] > np.pi*2/3:
                            p2[4] -= np.pi*2/3
                        p1[3], p2[4] = p2[4], p1[3]
                    else:
                        if p1[4] > np.pi*2/3:
                            p1[4] -= np.pi*2/3
                        p1[4], p2[3] = p2[3], p1[4]


        print(index)
