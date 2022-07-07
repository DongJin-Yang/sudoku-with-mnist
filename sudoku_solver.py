# # 스도쿠 풀이 알고리즘
# from main import sudoku, zeros, flag, result

# def is_promising(i, j):
#     global sudoku

#     promising = [1,2,3,4,5,6,7,8,9]  
    
#     #행열 검사
#     for k in range(9):
#         if sudoku[i][k] in promising:
#             promising.remove(sudoku[i][k])
#         if sudoku[k][j] in promising:
#             promising.remove(sudoku[k][j])
    
#     #3*3 박스 검사
#     i //= 3
#     j //= 3
#     for p in range(i*3, (i+1)*3):
#         for q in range(j*3, (j+1)*3):
#             if sudoku[p][q] in promising:
#                 promising.remove(sudoku[p][q])
#     return promising

# def dfs(x):
#     global sudoku, zeros, flag, result
    
#     if flag: #이미 답이 출력된 경우
#         return
        
#     if x == len(zeros): #마지막 0까지 다 채웠을 경우
#         flag = True #답 출력
#         for row in sudoku:
#             result.append(row.copy())
#         return 
        
#     else:    
#         (i, j) = zeros[x]
#         promising = is_promising(i, j) #유망한 숫자들을 받음
        
#         for num in promising:
#             sudoku[i][j] = num #유망한 숫자 중 하나를 넣어줌
#             dfs(x + 1) #다음 0으로 넘어감
#             sudoku[i][j] = 0 #초기화 (정답이 없을 경우를 대비)
