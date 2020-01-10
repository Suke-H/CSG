import numpy as np
import cv2 as cv

from method2d import *
from MakeDataset import MakePointSet

def TransPix(points, x_pix=1000):
    # AABBの横長dx, 縦長dyを出す
    max_p, min_p, _, _, _ = buildAABB(points)
    dx = abs(max_p[0] - min_p[0])
    dy = abs(max_p[1] - min_p[1])

    # 画像ではxを1000に固定する -> yはint(dx/dy*1000)
    px, py = x_pix, int(dx/dy*x_pix)

    print("px, py = {}, {}".format(px, py))

    # 原点はmin_p
    cx, cy = min_p

    # pixel内に点があれば255, なければ0の画像作成
    pix = np.zeros((py, px))
    X, Y = Disassemble2d(points)

    for i in range(py):
        print(i)
        for j in range(px):
            x1 = (cx + dx/px*j <= X).astype(int)
            x2 = (X <= cx + dx/px*(j+1)).astype(int)
            y1 = (cy + dy/py*i <= Y).astype(int)
            y2 = (Y <= cy + dy/py*(i+1)).astype(int)
            judge = x1 * x2 * y1 * y2

            if np.any(judge == 1):
                print("yeah!")
                #a = input()
                pix[i][j] = 255

    pix = np.array(pix, dtype=np.int)
    pix = np.flipud(pix)

    RGB_pix = np.array([pix, pix, pix])
    RGB_pix = np.transpose(RGB_pix, (1,2,0))

    cv.imwrite('data/test6.png', pix)
    cv.imwrite('data/test6rgb.png', RGB_pix)


def Morphology(path, dilate_size=25, erode_size=10, open_size=30, close_size=25):
     # ファイルを読み込み
    #pix = cv.imread(path, cv.IMREAD_GRAYSCALE)
    pix = cv.imread(path, cv.IMREAD_COLOR)
    # 画像の大きさ取得
    height, width, _ = pix.shape
    image_size = height * width

    # グレースケール変換
    dst = cv.cvtColor(pix, cv.COLOR_RGB2GRAY)

    # モルフォロジー
    dilate_kernel = np.ones((dilate_size,dilate_size),np.uint8)
    dst = cv.dilate(dst, dilate_kernel, iterations = 1)
    cv.imwrite('data/dilRGB.png', dst)

    # erode_kernel = np.ones((erode_size,erode_size),np.uint8)
    # dst = cv2.erode(pix,erode_kernel,iterations = 1)
    # cv.imwrite('data/erodeRGB.png', dst)

    close_kernel = np.ones((close_size, close_size),np.uint8)
    dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, close_kernel)
    cv.imwrite('data/closeRGB.png', dst)

    open_kernel = np.ones((open_size,open_size),np.uint8)
    dst = cv.morphologyEx(dst, cv.MORPH_OPEN, open_kernel)
    cv.imwrite('data/openRGB.png', dst)

    # 輪郭を抽出
    dst, contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print(contours)

    # この時点での状態をデバッグ出力
    dst = cv.imread(path, cv.IMREAD_COLOR)
    after_pix = cv.drawContours(dst, contours, -1, (0, 0, 255, 255), 2, cv.LINE_AA)
    cv.imwrite('data/afterRGB.png', dst)


# _, sign, _ = MakePointSet(2, 500)
# # 点群プロット
# X1, Y1= Disassemble2d(sign)
# plt.plot(X1, Y1, marker="o",linestyle="None",color="orange")
# plt.show()

# TransPix(sign)

Morphology("data/test6rgb.png")

