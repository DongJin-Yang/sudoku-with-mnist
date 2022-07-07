import cv2
import numpy as np
import imutils
from find_board import *
from perspective_warp import *
from number_list import *
# from sudoku_solver import 

from keras.models import load_model


model = load_model('model.h5')

img = cv2.imread('sudoku_p1.jpg')


board, approx = find_board(img)
p_board = perspective_warp(board, approx)
sudoku = number_list(p_board)

zeros = [(i, j) for i in range(9) for j in range(9) if sudoku[i][j] == 0]
flag = False #답이 출력되었는가?
result = []

# 스도쿠 풀이 알고리즘

def is_promising(i, j):
    promising = [1,2,3,4,5,6,7,8,9]  
    
    #행열 검사
    for k in range(9):
        if sudoku[i][k] in promising:
            promising.remove(sudoku[i][k])
        if sudoku[k][j] in promising:
            promising.remove(sudoku[k][j])
    
    #3*3 박스 검사
    i //= 3
    j //= 3
    for p in range(i*3, (i+1)*3):
        for q in range(j*3, (j+1)*3):
            if sudoku[p][q] in promising:
                promising.remove(sudoku[p][q])
    return promising

flag = False #답이 출력되었는가?
result = []
def dfs(x):
    global flag, result
    
    if flag: #이미 답이 출력된 경우
        return
        
    if x == len(zeros): #마지막 0까지 다 채웠을 경우
        flag = True #답 출력
        for row in sudoku:
            result.append(row.copy())
        return 
        
    else:    
        (i, j) = zeros[x]
        promising = is_promising(i, j) #유망한 숫자들을 받음
        
        for num in promising:
            sudoku[i][j] = num #유망한 숫자 중 하나를 넣어줌
            dfs(x + 1) #다음 0으로 넘어감
            sudoku[i][j] = 0 #초기화 (정답이 없을 경우를 대비)


dfs(0)

if not result:
    print('잘못된 스도쿠\n', sudoku)
else:
    print(result)