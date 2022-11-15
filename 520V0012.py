import cv2
import numpy as np

img = cv2.imread('input1.jpg')

yellow = np.uint8([[[0, 255, 255]]])
orange = np.uint8([[[20, 105, 229]]])
green = np.uint8([[[60, 154, 0]]])
red = np.uint8([[[83, 0, 223]]])
blue = np.uint8([[[193, 113, 0]]])
purple = np.uint8([[[123, 32, 111]]])

colorList = [yellow, orange, red, blue, green, purple]

# differences of values between green and the other colors
# used for creating 
diffColor = [np.array([45, 0, 0], dtype=np.uint8),
             np.array([45, 0, 0], dtype=np.uint8),
             np.array([60, 0, 0], dtype=np.uint8),
             np.array([-130, 0, 0], dtype=np.uint8),
             np.array([0, 0, 0], dtype=np.uint8),
             np.array([-70, 0, 0], dtype=np.uint8)]


def show(img, name=None):
    cv2.imshow(name, img)
    cv2.waitKey()


def extractStar(img, bgrColor):
    # cvt to HSV img
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # cvt color: BGR to HSV
    hsvColor = cv2.cvtColor(bgrColor, cv2.COLOR_BGR2HSV)

    # calc lower and upper HSV
    maskColor = cv2.inRange(hsvImg, np.subtract(
        hsvColor, [[[10, 100, 100]]]), np.add(hsvColor, [[[10, 255, 255]]]))

    # extract
    res = cv2.bitwise_and(img, img, mask=maskColor)

    # reduce noise
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, (5, 5))

    # return pair of image of each star and its mask range
    return [res, maskColor]


def toGreen(img, color, diff):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsvColor = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsvImg, np.subtract(
        hsvColor, [[[10, 100, 100]]]), np.add(hsvColor, [[[10, 255, 255]]]))

    hsvImg[mask > 0] += diff
    hsvImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    return hsvImg


def toAllGreen(img, colorList, diffList):
    hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for i in range(len(colorList)):
        # calc mask range
        hsvColor = cv2.cvtColor(colorList[i], cv2.COLOR_BGR2HSV)
        maskColor = cv2.inRange(hsvImg, np.subtract(
            hsvColor, [[[10, 100, 100]]]), np.add(hsvColor, [[[10, 255, 255]]]))
        
        # replace range by relative diffColor
        hsvImg[maskColor > 0] += diffList[i]
    hsvImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    return hsvImg


def blackBorder():
    imgGray = cv2.imread('input1.jpg', 0)
    ret, th = cv2.threshold(imgGray, 220, 255, cv2.THRESH_TOZERO_INV)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, (3, 3))
    # show(th)
    return th


def blackStar():
    imgGray = cv2.imread('input1.jpg', 0)
    # src >= 190 ==> src = 0
    ret, th = cv2.threshold(imgGray, 190, 0, cv2.THRESH_TOZERO_INV)
    
    # scr > 180 => src=255, else src=0
    ret, th = cv2.threshold(th, 180, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, (5, 5))
    # show(th)
    return th


def sharpen(img):
    # smoothed = cv2.GaussianBlur(img, (9, 9), 10)
    # unsharped = cv2.addWeighted(img, 1.5, smoothed, -0.5, 0)
    # show(unsharped)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    imgSharped = cv2.filter2D(img, ddepth=-1, kernel=kernel)
    imgSharped = cv2.putText(imgSharped, '520V0012', (346, 326),
                             cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 0), thickness=2)
    # show(imgSharped)
    return imgSharped


def contourStar(img):
    img1 = np.copy(img)
    imgGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(imgGray, 220, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    # reduce noise
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, hierarchy = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(img1, [cnt], -1, (0, 0, 255), 2)
    return img1


def contourRec(img):
    img1 = np.copy(img)
    imgGray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(imgGray, 220, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)

    # reduce noise
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # find contours
    contours, hierarchy = cv2.findContours(
        th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img1


if __name__ == "__main__":
    starImgs = [extractStar(img, i) for i in colorList]
    greenStars = [toGreen(img, colorList[i], diffColor[i]) for i in range(6)]
    # allGreenStar = toAllGreen(img, colorList, diffColor)

    for i in range(len(starImgs)):
        # show(i[0])
        cv2.imwrite('star_' + str(i+1) + ".jpg", starImgs[i][0])
    for i in range(len(greenStars)):
        # show(i)
        cv2.imwrite('green_star_' + str(i+1) + ".jpg", greenStars[i])

    cv2.imwrite('all_green_stars.jpg', toAllGreen(img, colorList, diffColor))
    # show(allGreenStar)
    cv2.imwrite('black_borders.jpg', blackBorder())
    cv2.imwrite('black_stars.jpg', blackStar())
    cv2.imwrite('contour_stars.jpg', contourStar(img))
    # show(contourStar(img))
    cv2.imwrite('contour_rec_stars.jpg', contourRec(img))
    # show(contourRec(img))
    cv2.imwrite('sharped_image.jpg', sharpen(img))

    cv2.destroyAllWindows()
