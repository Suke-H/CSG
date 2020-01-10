import cv2 as cv
import numpy as np

def main():
    # ファイルを読み込み
    image_file = 'data/yoki2.png'
    src = cv.imread(image_file, cv.IMREAD_COLOR)
    # 画像の大きさ取得
    height, width, channels = src.shape
    image_size = height * width
    # グレースケール化
    img_gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    # しきい値指定によるフィルタリング
    retval, dst = cv.threshold(img_gray, 127, 255, cv.THRESH_TOZERO_INV )


    # 白黒の反転
    #dst = cv.bitwise_not(dst)
    # 再度フィルタリング
    retval, dst = cv.threshold(dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    cv.imwrite('data/before.png', dst)

    # モルフォロジー
    kernel = np.ones((7,7),np.uint8)
    dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, kernel)
    cv.imwrite('data/closing.png', dst)
    kernel = np.ones((10,10),np.uint8)
    dil = cv.dilate(dst, kernel, iterations = 1)
    cv.imwrite('data/dilation.png', dil)
    opening = cv.morphologyEx(dst, cv.MORPH_OPEN, kernel)
    cv.imwrite('data/opening.png', opening)

    # 輪郭を抽出
    dst, contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    print(contours)
    
    # この時点での状態をデバッグ出力
    dst = cv.imread(image_file, cv.IMREAD_COLOR)
    dst = cv.drawContours(dst, contours, -1, (0, 0, 255, 255), 2, cv.LINE_AA)
    cv.imwrite('data/debug_4.png', dst)
    dst = cv.imread(image_file, cv.IMREAD_COLOR)
    for i, contour in enumerate(contours):
        # 小さな領域の場合は間引く
        area = cv.contourArea(contour)
        if area < 500:
            continue
        # 画像全体を占める領域は除外する
        if image_size * 0.99 < area:
            continue
        
        # 外接矩形を取得
        x,y,w,h = cv.boundingRect(contour)
        dst = cv.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
    # 結果を保存
    cv.imwrite('data/result4.png', dst)

if __name__ == '__main__':
    main()