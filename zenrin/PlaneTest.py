from GAtest import *

def test3D(fig_type, loop):
    count = 0

    while count != loop:
        # 2d図形点群作成
        para3d, sign3d, AABB3d, trueIndex = MakePointSet3D(fig_type, 500, rate=0.8)

        # 平面検出, 2d変換
        sign2d, plane, u, v, O, index1 = PlaneDetect(sign3d)

        # 外枠作成
        out_points, out_area = MakeOuterFrame(sign2d, path="data/GAtest/" + str(count) + ".png")

        # 外枠内の点群だけにする
        index2 = np.array([CheckClossNum3(sign2d[i], out_points) for i in range(sign2d.shape[0])])
        #inside = CheckClossNum2(sign2d, out_points)
        sign2d = sign2d[index2]

        # GAにより最適パラメータ出力
        #best = GA(sign)
        best = EntireGA(sign2d, out_points, out_area)
        print("="*50)

        X, Y = Disassemble2d(sign2d)
        index3 = (best.figure.f_rep(X, Y) >= 0)

        estiIndex = SelectIndex(index1, SelectIndex(index2, index3))

        print(best[fig_type].figure.p)

        count+=1