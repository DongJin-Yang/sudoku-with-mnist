import cv2
import numpy as np

def perspective_warp(board, approx):
    # board의 꼭지점 찾기
    pts = approx.reshape(4, 2)
    # for x, y in pts:
    #     cv2.circle(draw, (x, y), 10, (0, 255, 0), -1)
    sm = pts.sum(axis = 1)
    diff = np.diff(pts, axis = 1)

    topLeft = pts[np.argmin(sm)]
    bottomRight = pts[np.argmax(sm)]
    topRight = pts[np.argmin(diff)]
    bottomLeft = pts[np.argmax(diff)]
    
    # pts1 점을 pts2 점으로 이동
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])
    height = max([h1, h2])

    pts2 = np.float32([[0, 0], [width-1, 0], [width -1, height-1], [0, height-1]])
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(board, mtrx, (width, height))
    
    # 한 칸의 길이를 28로 정함
    result = cv2.resize(result, (28*9, 28*9))
    
    
#     cv2.imshow('result', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    return result