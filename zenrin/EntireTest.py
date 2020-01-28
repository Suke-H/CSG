from GAtest import *
import open3d

def test3D(fig_type, num, out_path):

    if fig_type != 2:
        rec_list = ["pos", "size", "angle", "TP", "TN", "FP", "FN", "acc", "prec", "rec", "F_measure"]

    else:
        rec_list = ["pos", "size1", "size2", "angle", "TP", "TN", "FP", "FN", "acc", "prec", "rec", "F_measure"]

    with open(out_path+"test.csv", 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rec_list)

    for i in range(num):
        # 2d図形点群作成
        center, para2d, plane_para, points3d, AABB, trueIndex = MakePointSet3D(fig_type, 500, rate=0.8)

        max_p, min_p, l = buildAABB(points3d)

        # 法線推定
        normals = NormalEstimate(points3d)

        # 平面検出, 2d変換
        points2d, plane, u, v, O, index1 = PlaneDetect(points3d, normals, epsilon=0.01*l, alpha=np.pi/12)

        opti_plane_para = plane.p

        # 外枠作成
        out_points, out_area = MakeOuterFrame2(points2d, out_path, i, 
                                dilate_size1=30, close_size1=20, open_size1=50, add_size1=50,
                                dilate_size2=30, close_size2=0, open_size2=50, add_size2=5, goalDensity=10000)

        # 外枠内の点群だけにする
        index2 = np.array([CheckClossNum3(points2d[i], out_points) for i in range(points2d.shape[0])])
        #inside = CheckClossNum2(sign2d, out_points)
        points2d = points2d[index2]

        # GAにより最適パラメータ出力
        #best = GA(sign)
        best = EntireGA(points2d, out_points, out_area, CalcIoU1, out_path+"/GA/"+str(i)+".png", 
                        add_num=30, half_reset_num=15, all_reset_num=9)

        # 検出図形の中心座標を3次元に射影
        opti_para2d = best.figure.p
        opti_center, _ = Plane3DProjection(points2d, opti_para2d, u, v, O)
        
        #######################################################################

        rec_list = []

        # 位置: 3次元座標上の中心座標の距離
        pos = LA.norm(center - opti_center)
        rec_list.append(pos)

        # 大きさ: 円と三角形ならr, 長方形なら長辺と短辺
        if fig_type != 2:
            size = abs(para2d[2] - opti_para2d[2])
            rec_list.append(size)

        else:
            if para2d[2] > para2d[3]:
                long_edge, short_edge = para2d[2], para2d[3]
            else:
                long_edge, short_edge = para2d[3], para2d[2]

            if opti_para2d[2] > opti_para2d[3]:
                opti_long_edge, opti_short_edge = opti_para2d[2], opti_para2d[3]
            else:
                opti_long_edge, opti_short_edge = opti_para2d[3], opti_para2d[2]

            size1 = abs(long_edge - opti_long_edge)
            size2 = abs(short_edge - opti_short_edge)
            rec_list.append(size1)
            rec_list.append(size2)

        # 平面の法線の角度
        n_goal = np.array([plane_para[0], plane_para[1], plane_para[2]])
        n_opt = np.array([opti_plane_para[0], opti_plane_para[1], opti_plane_para[2]])
        angle = np.arccos(np.dot(n_opt, n_goal))
        rec_list.append(angle)

        # 形: 混合行列で見る
        X, Y = Disassemble2d(points2d)
        index3 = (best.figure.f_rep(X, Y) >= 0)

        print(index1.shape, np.count_nonzero(index1))
        print(index2.shape, np.count_nonzero(index2))
        print(index3.shape, np.count_nonzero(index3))

        estiIndex = SelectIndex(index1, SelectIndex(index2, index3))

        print(estiIndex.shape, np.count_nonzero(estiIndex))

        confusionIndex = ConfusionLabeling(trueIndex, estiIndex)

        TP = np.count_nonzero(confusionIndex==1)
        TN = np.count_nonzero(confusionIndex==2)
        FP = np.count_nonzero(confusionIndex==3)
        FN = np.count_nonzero(confusionIndex==4)

        acc = (TP+TN)/(TP+TN+FP+FN)
        prec = TP/(TP+FN)
        rec = TP/(TP+FP)
        F_measure = 2*prec*rec/(prec+rec)

        rec_list.extend([TP, TN, FP, FN, acc, prec, rec, F_measure])

        with open(out_path+"test.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(rec_list)

        ViewTest(points3d, confusionIndex)



def ViewTest(sign3d, confusionIndex):
    #点群をnp配列⇒open3d形式に
        pointcloud = open3d.PointCloud()
        pointcloud.points = open3d.Vector3dVector(sign3d)

        # color
        # colors3 = np.asarray(pointcloud.colors)
        num = sign3d.shape[0]
        
        colors3 = []

        for i in range(num):
            if confusionIndex[i] == 1:
                colors3.append([255, 0, 0])
            elif confusionIndex[i] == 2:
                colors3.append([0, 0, 255])
            elif confusionIndex[i] == 3:
                colors3.append([255, 130, 0])
            else:
                colors3.append([0, 255, 255])

        colors3 = np.array(colors3) / 255
        pointcloud.colors = open3d.Vector3dVector(colors3)

        # 法線推定
        open3d.estimate_normals(
        pointcloud,
        search_param = open3d.KDTreeSearchParamHybrid(
        radius = 10, max_nn = 100))

        # 法線の方向を視点ベースでそろえる
        open3d.orient_normals_towards_camera_location(
        pointcloud,
        camera_location = np.array([0., 10., 10.], 
        dtype="float64"))

        #nキーで法線表示
        open3d.draw_geometries([pointcloud])


test3D(1, 1, out_path="data/EntireTest/test/")