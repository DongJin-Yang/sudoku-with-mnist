import cv2
import imutils


def find_board(img): 
    # 기존의 이미지를 바이너리 이미지로 만듦
    draw = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary_image = cv2.bitwise_not(thresh, thresh)
    
    # board의 외각선 범위만 추출
    cnts = cv2.findContours(binary_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse = True)[:5]
    
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
            
    cv2.drawContours(draw, [screenCnt], -1, (0, 255, 0), 2)
    
    
#     cv2.imshow('draw', thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    return binary_image, approx