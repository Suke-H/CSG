import numpy as np
import copy
import matplotlib.pyplot as plt
import csv

from method2d import *
import figure2d as F
from IoUtest import CalcIoU, CalcIoU2

class person:
    def __init__(self, fig_type, figure):
        self.fig_type = fig_type
        self.figure = figure
        self.score = 0
        self.scoreFlag = False
        self.area = figure.CalcArea()

def EntireGA(points, fig=[0,1,2], n_epoch=1000, N=100, add_num=30, save_num=2, tournament_size=10, \
    mutate_rate=1, path=None):
    # AABB生成
    max_p, min_p, _, l, _ = buildAABB(points)

    # 図形の種類ごとにN人クリーチャー作成
    group = np.array([CreateRandomPopulation(N, max_p, min_p, l, fig[i]) for i in range(len(fig))])
    print(group.shape)

    for epoch in range(n_epoch):
        #print("epoch:{}".format(epoch))
        # 新しいクリーチャー追加
        group = np.array([np.concatenate([group[i], CreateRandomPopulation(add_num, max_p, min_p, l, fig[i])]) for i in range(len(fig))])
        # スコア順に並び替え
        group = np.array([Rank(group[i], points)[0] for i in range(len(fig))])
        # csv?に保存
        if path:
            SaveCSV(group[0][0], epoch, path)
        # 上位n人は保存
        next_group = np.array([group[i][:save_num] for i in range(len(fig))])

        # 次世代がN人超すまで
        # トーナメント選択->交叉、突然変異->保存
        # を繰り返す
        while len(next_group[0]) <= N:
            # トーナメントサイズの人数出場
            entry = np.array([np.random.choice(group[i], tournament_size, replace=False) for i in range(len(fig))])
            #print(entry.shape)

            cross_children, mutate_children = [], []

            for i, f in enumerate(fig):
                # 図形の種類によって"Crossover"で取る個体の数変更
                if f == 0:
                    num = 4
                elif f == 1:
                    num = 5
                elif f == 2:
                    num = 6

                c = None

                while c is None:

                    # 上位num+1人選択
                    entry_tmp, _ = Rank(entry[i], points)[:num+1]
                    # 突然変異させる人を選択
                    mutate_index = np.random.choice(num+1)
                    # それ以外を交叉
                    #cross_children.append([Crossover(np.delete(entry_tmp, mutate_index))])
                    c = Crossover2(np.delete(entry_tmp, mutate_index), f, max_p, min_p, l)
                    if c is not None:
                        cross_children.append([c])
                    # 突然変異
                    mutate_children.append([Mutate(entry_tmp[mutate_index], max_p, min_p, l, rate=mutate_rate)])
                
            
            #print(next_group.shape, cross_group.shape, mutate_group.shape)
            #print(len(next_group), len(cross_children), len(mutate_children))
            
            # c1, c2, c3を次世代に追加
            next_group = np.concatenate([next_group, cross_children, mutate_children], axis=1)
            #print("next:{}".format(next_group.shape))

        group = next_group[:, :]

        
        # 途中経過表示
        if epoch % 30 == 0:
            print("{}回目成果".format(int(epoch/30)))

            for i in range(len(fig)):
                _, score_list = Rank(group[i], points)
                print(score_list[:10])
                print(group[i][0].figure.p)
                #DrawFig(points, group[i][0])
                #DrawFig(points, people[1])
        
    # 最終結果表示
    for i in range(len(fig)):
        _, score_list = Rank(group[i], points)
        print(score_list[:10])
        DrawFig(points, group[i][0])

    return [group[i][0].figure.p for i in range(len(fig))]

def GA(points, fig=[0,1,2], epoch=100, N=500, add_num=50, save_num=1, tournament_size=50, cross_rate=0.75, mutate_rate=0.2):
    # AABB生成
    max_p, min_p, _, l, _ = buildAABB(points)

    # N人クリーチャー作成
    people = CreateRandomPopulation(N, max_p, min_p, l, fig)

    for i in range(epoch):
        #print("epoch:{}".format(i))
        # 新しいクリーチャー追加
        people = np.concatenate([people, CreateRandomPopulation(add_num, max_p, min_p, l, fig)])
        # スコア順に並び替え
        people, _ = Rank(people, points)
        # 上位n人は保存
        next_people = people[:save_num]

        # 次世代がN人超すまで
        # トーナメント選択->交叉、突然変異->保存
        # を繰り返す
        while len(next_people) <= N:
            # トーナメントサイズの人数出場
            entry = np.random.choice(people, tournament_size, replace=False)
            # 上位3人選択
            entry, _ = Rank(entry, points)[:3]
            # 1位と2位を交叉
            c1, c2 = Crossover(entry[0], entry[1], rate=cross_rate)
            # 3位を突然変異
            c3 = Mutate(entry[2], max_p, min_p, l, rate=mutate_rate)
            
            next_people = np.append(next_people, c1)
            next_people = np.append(next_people, c2)
            next_people = np.append(next_people, c3)

        people = next_people[:]

        # 途中経過表示
        if i % 10 == 0:
            sorted_People, score_list = Rank(people, points)
            print("{}回目成果".format(int(i/10)))
            print(score_list[:10])
            DrawFig(points, people[0])
            #DrawFig(points, people[1])
            

    # 最終結果表示
    sorted_People, score_list = Rank(people, points)
    print(score_list[:10])
    DrawFig(points, people[0])
    #DrawFig(points, people[1])

    return people[0]

def CreateRandomPerson(fig_type, max_p, min_p, l):
    # 円
    if fig_type==0:
        #print("円")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < r < 2/3*l
        r = Random(l/10, 2/3*l)

        figure = F.circle([x,y,r])

    # 正三角形
    elif fig_type==1:
        #print("正三角形")
        # min_p < x,y < max_p
        x = Random(min_p[0], max_p[0])
        y = Random(min_p[1], max_p[1])
        # 0 < r < 2/3*l
        r = Random(l/10, 2/3*l)
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
        w = Random(l/10, l)
        h = Random(l/10, l)
        # 0 < t < pi
        t = Random(0, np.pi)

        figure = F.rect([x,y,w,h,t])

    return person(fig_type, figure)
    
def CreateRandomPopulation(num, max_p, min_p, l, fig):
    # ランダムに図形の種類を選択し、遺伝子たちを生成
    population = np.array([CreateRandomPerson(fig, max_p, min_p, l) for i in range(num)])

    return population

def Score(person, points):
    # scoreFlagが立ってなかったらIoUを計算
    if person.scoreFlag == False:
        #person.score = CalcIoU(points, person.figure)
        person.score = CalcIoU2(points, person.figure)
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
def Mutate(person, max_p, min_p, l, rate=1.0):
    # rateの確率で突然変異
    if np.random.rand() <= rate:
        #personに直接書き込まないようコピー
        person = copy.deepcopy(person)
        # 図形パラメータの番号を選択
        index = np.random.choice([i for i in range(len(person.figure.p))])
        # 図形の種類にそって、選択したパラメータをランダムに変更

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
                person.figure.p[index] = Random(l/10, 2/3*l)

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
                person.figure.p[index] = Random(l/10, 2/3*l)
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
                person.figure.p[index] = Random(l/10, l)
            # h
            elif index == 3:
                person.figure.p[index] = Random(l/10, l)
            # t
            else:
                person.figure.p[index] = Random(0, np.pi/2)

    return person

# 同じ図形同士なら、場所を選択して交叉
# [x1,y1,r1][x2,y2,r2] -> 1を選択 -> [x1,y2,r2][x2,y1,r1]

# 違う図形同士なら、共通するパラメータを1つだけ交換
# [x1,y1,r][x2,y2,w,h,t] -> [x,y]が共通 -> 1を選択 -> [x1,y2,r][x2,y1,w,h,t]
# [x1,y1,r,t1][x2,y2,w,h,t2] -> [x,y,t]が共通 -> 2を選択 -> [x1,y1,r,t2][x2,y2,w,h,t1]
def Crossover(person1, person2, rate=1.0):
    # rateの確率で突然変異
    if np.random.rand() <= rate:
        # personに直接書き込まないようコピー
        person1, person2 = copy.deepcopy(person1), copy.deepcopy(person2)

        f1, f2, p1, p2 = person1.fig_type, person2.fig_type, person1.figure.p, person2.figure.p

        # 同じ図形なら
        if f1 == f2:
            # 図形パラメータの番号を選択
            index = np.random.choice([i for i in range(len(p1))])
            # 同じ番号の場所を交代
            p1[index], p2[index] = p2[index], p1[index]

        # 違う図形なら
        else:
            print("error")
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

    return person1, person2

# BLX-a
def BLX(x1, x2, xmin, xmax, alpha):
    r = Random(-alpha, 1+alpha)
    x = r*x1 + (1-r)*x2

    if any(xmin < x) and any(x < xmax):
        return x
        
    else:
        return BLX(x1, x2, xmin, xmax, alpha)

# ブレンド交叉を採用
def Crossover2(parents, fig, max_p, min_p, l):

    while True:

        # n: パラメータの数, x: n+1人の親のパラメータのリスト
        n = len(parents[0].figure.p)
        x = np.array([parents[i].figure.p for i in range(n+1)])

        # g: xの重心
        g = np.sum(x, axis=0) / n

        alpha = np.sqrt(n+2)

        # p, cを定義
        p, c = np.empty((0,n)), np.empty((0,n))
        p = np.append(p, [g + alpha*(x[0] - g)], axis=0)
        c = np.append(c, [[0 for i in range(n)]], axis=0)

        for i in range(1, n+1):
            r = Random(0, 1)**(1/i)
            p = np.append(p, [g + alpha*(x[i] - g)], axis=0)
            c = np.append(c, [r*(p[i-1]-p[i] + c[i-1])], axis=0)
            #print(r, p[i], c[i])

        # 子のパラメータはp[n]+c[n]となる
        child = p[n] + c[n]

        # パラメータが範囲外ならやり直し
        if CheckIB(child, fig, max_p, min_p, l):
            break

    # パラメータが範囲外なら子は生成しない
    # if not CheckIB(child, fig, max_p, min_p, l):
    #     return None
        
    # パラメータをpersonクラスに代入する
    fig = parents[0].fig_type

    if fig == 0:
        figure = F.circle(child)
    elif fig == 1:
        figure = F.tri(child)
    elif fig == 2:
        figure = F.rect(child)

    return person(fig, figure)

def CheckIB(child, fig, max_p, min_p, l):
    # 円
    if fig==0:
        x, y, r = child
        w, h, t_tri, t_rec = l/2, l/2, np.pi/6, np.pi/6
    # 正三角形
    elif fig==1:
        x, y, r, t_tri = child
        w, h, t_rec = l/2, l/2, np.pi/6
    # 長方形
    elif fig==2:
        x, y, w, h, t_rec = child
        r, t_tri = l/2, np.pi/6

    if (min_p[0] < x < max_p[0]) and (min_p[1] < y < max_p[1]) and (0 < r < l) and (0 < w < l) and (0 < h < l):
        #and (0 < t_tri < np.pi*2/3) and (0 < t_rec < np.pi/2)
        return True

    else:
        return False

            

def DrawFig(points, person):
    # 目標点群プロット
    X1, Y1= Disassemble2d(points)
    plt.plot(X1, Y1, marker=".",linestyle="None",color="yellow")

    # 推定図形プロット
    max_p, min_p, _, _, _ = buildAABB(points)
    AABB = [min_p[0], max_p[0], min_p[1], max_p[1]]
    points2 = ContourPoints(person.figure.f_rep, AABB=AABB, grid_step=1000, epsilon=0.01, down_rate = 0.5)
    X2, Y2= Disassemble2d(points2)
    plt.plot(X2, Y2, marker="o",linestyle="None",color="red")

    plt.show()

def SaveCSV(person, epoch, path):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, person.score])
