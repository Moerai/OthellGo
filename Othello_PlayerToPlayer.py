import os, copy
import random
import time

n = 8  # board size (even)
board = [['0' for x in range(n)] for y in range(n)]
# 8 directions
dirx = [-1, 0, 1, -1, 1, -1, 0, 1]
diry = [-1, -1, -1, 0, 0, 1, 1, 1]


def InitBoard():
    if n % 2 == 0:  # if board size is even
        z = int((n - 2) / 2)
        board[z][z] = '2'
        board[n - 1 - z][z] = '1'
        board[z][n - 1 - z] = '1'
        board[n - 1 - z][n - 1 - z] = '2'


def PrintBoard():
    m = len(str(n - 1))
    for y in range(n):
        row = ''
        for x in range(n):
            row += board[y][x]
            row += ' ' * m
        print(row + ' ' + str(y))
    print
    row = ''
    for x in range(n):
        row += str(x).zfill(m) + ' '
    print(row + '\n')


def MakeMove(board, x, y, player):  # assuming valid move
    totctr = 0  # total number of opponent pieces taken
    board[y][x] = player
    for d in range(8):  # 8 directions
        ctr = 0
        for i in range(n):
            dx = x + dirx[d] * (i + 1)
            dy = y + diry[d] * (i + 1)
            if dx < 0 or dx > n - 1 or dy < 0 or dy > n - 1:
                ctr = 0;
                break
            elif board[dy][dx] == player:
                break
            elif board[dy][dx] == '0':
                ctr = 0;
                break
            else:
                ctr += 1
        for i in range(ctr):
            dx = x + dirx[d] * (i + 1)
            dy = y + diry[d] * (i + 1)
            board[dy][dx] = player
        totctr += ctr
    return (board, totctr)


def ValidMove(board, x, y, player):
    if x < 0 or x > n - 1 or y < 0 or y > n - 1:
        return False
    if board[y][x] != '0':
        return False
    (boardTemp, totctr) = MakeMove(copy.deepcopy(board), x, y, player)
    if totctr == 0:
        return False
    return True


minEvalBoard = -1  # min - 1
maxEvalBoard = n * n + 4 * n + 4 + 1  # max + 1


def EvalBoard(board, player):
    tot = 0
    for y in range(n):
        for x in range(n):
            if board[y][x] == player:
                # if (x == 0 or x == n - 1) and (y == 0 or y == n - 1):
                #     tot += 4  # corner
                # elif (x == 0 or x == n - 1) or (y == 0 or y == n - 1):
                #     tot += 2  # side
                # else:
                tot += 1
    return tot


def IsTerminalNode(board, player):
    for y in range(n):
        for x in range(n):
            if ValidMove(board, x, y, player):
                return False
    return True


def GetSortedNodes(board, player):
    sortedNodes = []
    for y in range(n):
        for x in range(n):
            if ValidMove(board, x, y, player):
                (boardTemp, totctr) = MakeMove(copy.deepcopy(board), x, y, player)
                sortedNodes.append((boardTemp, EvalBoard(boardTemp, player)))
    sortedNodes = sorted(sortedNodes, key=lambda node: node[1], reverse=True)
    sortedNodes = [node[0] for node in sortedNodes]
    return sortedNodes


InitBoard()
while True:
    for p in range(2):
        print
        # PrintBoard()
        player = str(p + 1)
        print('PLAYER: ' + player)
        if IsTerminalNode(board, player):
            print('Player cannot play! Game ended!')
            print('Score Comp1: ' + str(EvalBoard(board, '1')))
            print('Score Comp2: ' + str(EvalBoard(board, '2')))
            os._exit(0)
        # if player == '1':  # user's turn
        #         #     while True:
        #         #         xy = input('X Y: ')
        #         #         if xy == '': os._exit(0)
        #         #         (x, y) = xy.split()
        #         #         x = int(x)
        #         #         y = int(y)
        #         #         if ValidMove(board, x, y, player):
        #         #             (board, totctr) = MakeMove(board, x, y, player)
        #         #             print('# of pieces taken: ' + str(totctr))
        #         #             break
        #         #         else:
        #         #             print('Invalid move! Try again!')
        #         #     if not (x == -1 and y == -1):
        #         #         (board, totctr) = MakeMove(board, x, y, player)
        #         #         print('player1 played (X Y): ' + str(x) + ' ' + str(y))
        #         #         print('# of pieces taken: ' + str(totctr))
        #         #         PrintBoard()
        if player == '1':  # computer1's turn
            while True:
                validlist = []
                for x in range(8):
                    for y in range(8):
                        if ValidMove(board, x, y, player):
                            validlist.append([x, y])
                random.shuffle(validlist)
                (x, y) = validlist.pop()
                x = int(x)
                y = int(y)
                (board, totctr) = MakeMove(board, x, y, player)
                print('# of pieces taken: ' + str(totctr))
                break

            if not (x == -1 and y == -1):
                (board, totctr) = MakeMove(board, x, y, player)
                print('player1 played (X Y): ' + str(x) + ' ' + str(y))
                print('# of pieces taken: ' + str(totctr))
                PrintBoard()
                # time.sleep(1)

        else:  # computer2's turn
            while True:
                validlist = []
                for x in range(8):
                    for y in range(8):
                        if ValidMove(board, x, y, player):
                            validlist.append([x, y])
                random.shuffle(validlist)
                (x, y) = validlist.pop()
                x = int(x)
                y = int(y)
                (board, totctr) = MakeMove(board, x, y, player)
                print('# of pieces taken: ' + str(totctr))
                break

            if not (x == -1 and y == -1):
                # (board, totctr) = MakeMove(board, x, y, player)
                print('player2 played (X Y): ' + str(x) + ' ' + str(y))
                print('# of pieces taken: ' + str(totctr))
                PrintBoard()
