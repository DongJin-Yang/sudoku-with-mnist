import cv2
import numpy as np
from keras.models import load_model
model = load_model('model.h5')

def number_list(board):
    # board에서 한칸씩 순서대로 squares리스트에 저장
    size=28
    squares = []
    for y in range(1, 10):
        for x in range(1, 10):
            squares.append(board[(y-1)*size:y*size, (x-1)*size:x*size])
    
    # 칸의 테두리를 제외하고 내부의 분산을 확인해 빈칸이 아니면 numbered 리스트에 저장
    # 빈칸이라면 분산은 0이 된다.
    numbered = []
    for i in range(81):
        if np.var(squares[i][5:25, 5:25])>2000:    # [:, :] -> 범위 수정 코드 짜기
            numbered.append(i)
    
    # 숫자 예측
    my_dict = {}
    for index in numbered:
        buff = squares[index][3:25, 3:25]
        buff = cv2.resize(buff, (28, 28))
        buff = buff.reshape(1, 28, 28, 1)

        label = model.predict(buff)
        my_dict[index] = label[0]
    
    # board의 숫자들을 2차원 리스트로 저장
    sudoku = [0]*81
    for key, value in my_dict.items():
        num = value.tolist()
        sudoku[key] = num.index(1)
        
    sudoku = np.array(sudoku)
    sudoku = sudoku.reshape(9, 9)
    sudoku = sudoku.tolist()
        
    return sudoku
