import random
import sys

def drawBoard(board):
    HLINE = '  +---+---+---+---+---+---+---+---+'
    VLINE = '  |   |   |   |   |   |   |   |   |'
    print('    1   2   3   4   5   6   7   8')
    print(HLINE)
    for y in range(8):
        print(VLINE)
        print(y + 1, end=' ')
        for x in range(8):
            print('| %s' % (board[x][y]), end=' ')
        print('|')
        print(VLINE)
        print(HLINE)

def resetBoard(board):
    # Blanks out the board it is passed, except for the original starting positoin.
    for x in range(8):
        for y in range(8):
            board[x][y] == ' '

    # Starting pieces:
    board[3][3] = 'X'
    board[3][4] = 'O'
    board[4][3] = 'O'
    board[4][4] = 'X'

def getNewBoard():
    # Creates a brand new
    board = [] # board는 8*8
    for i in range(8):
        board.append([' ']*8)

    return board

def isValidMove(board, tile, xstart, ystart):
    # Returns False if the player's move in space xtart, ystart is invalid.
    if board[xstart][ystart] != ' ' or not isOnBoard(xstart, ystart):
        return False
    board[xstart][ystart] = tile
    if tile == 'X':
        otherTile = 'O'
    else:
        otherTile = 'X'
    tilesToFlip = [] # 움직임을 통해 뒤집힐 상대편 타일의 리스트를 반환

    for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
        x, y = xstart, ystart
        # 조사를 위해 각 방향으로 움직여 본다.
        x += xdirection
        y += ydirection
        # x,y가 유효하며, 이동한 위치에 상대방 타일이 있을 경우 -> 상대방 타일을 뒤집을 수 있는지 조사한다.
        if not isOnBoard(x,y):
            continue
        while board[x][y] == otherTile:
            x += xdirection
            y += ydirection
            # x,y가 유효하지 않으면 다음 for문
            if not isOnBoard(x,y):
                break
        if not isOnBoard(x, y):
            continue
        if board[x][y] == tile:
            while True:
                x -= xdirection
                y -= ydirection
                if x == xstart and y == ystart:
                    break
                tilesToFlip.append([x, y]) # 뒤집을 수 있는 타일을 계속 저장한다.

    board[xstart][ystart] == ' ' # restore the empty space
    if len(tilesToFlip) == 0: # If no tiles were flipped, this's not a valid move.
        return False
    return tilesToFlip

def isOnBoard(x,y):
    # Returns True if the coordinates are located on the board.
    return 0 <= x <= 7 and 0 <= y <= 7  # x,y가 좌표가 0~7범위안(보드판)위에 있는것.

def getBoardWithValidMoves(board, tile):
    # 플레이어에게 힌트를 주기위해 가능한 움직임을 board로 리턴한다.
    dupeBoard = getBoardCopy(board)

    for x, y in getValidMoves(dupeBoard, tile):
        dupeBoard[x][y] = '.'
    return dupeBoard

def getValidMoves(board, tile):
    # Returns a list of [x,y] lists of valid moves for the given player on the given board.
    validMoves = []
    for x in range(8):
        for y in range(8):
            if isValidMove(board, tile, x, y): # 가능한 움직임인지 isValidMove로 판별하고
                validMoves.append([x,y])
    return validMoves  # 가능한 위치를 보드로 리턴한다.

def getScoreOfBoard(board):
    # Determine the score by counting the tiles.
    xscore = 0
    oscore = 0
    for x in range(8):
        for y in range(8):
            if board[x][y] == 'X':
                xscore += 1
            if board[x][y] == 'O':
                oscore += 1
    return {'X': xscore, 'O': oscore}

def enterPlayerTile():
    # Lets the player type which tile they want to be.
    # Returns a list with the player's tile as the first item, and the computer's tile as the second.
    tile = ''
    while not (tile == 'X' or tile == 'O'):
        print('Do you want to be X or 0?')
        tile = input().upper()  # 플레이어의 타일을 고르도록한다.

    # the first element in the tuple is the player's tile, the second is the computer's tile.
    # 리스트로 반환
    if tile == 'X':
        return ['X', 'O']
    else:
        return ['O', 'X']

def whoGoesFirst():
    if random.randint(0, 1) == 0:
        return 'computer'
    else:
        return 'player'


def playAgain():
    print("Do you want to play again? (yes or no)")
    return input().lower().startswith('y') # input값을 소문자로 바꾸고 'y'로 시작하는지 판별.


def makeMove(board, tile, xstart, ystart):
    tilesToFlip = isValidMove(board, tile, xstart, ystart)  # 뒤집힐 수 있는 상대편의 타일 리스트

    if not tilesToFlip: # if tilesToFlip == False
        return False

    board[xstart][ystart] = tile
    for x, y in tilesToFlip:
        board[x][y] = tile  # 플레이어의 타일로 변경한다.
    return True

def getBoardCopy(board):
    # Make a duplicate of the board list and return the duplicate.
    dupeBoard = getNewBoard()  # 보드를 복사해온다.

    for x in range(8):
        for y in range(8):
            dupeBoard[x][y] = board[x][y]  # 복사본에 현 보드를 저장한다.
    return dupeBoard


def isOnCorner(x,y):
    # Returns True if the position is in one of the 4 corners.
    return (x == 0 and y == 0) or (x == 7 and y == 0) or (x == 0 and y == 7) and (x == 7 and y == 7)


def getPlayerMove(board, playerTile):
    DIGITS1TO8 = '1 2 3 4 5 6 7 8'.split()  # list ['1','2',...]
    while True:
        print("Enter you move, or type quit to end the game,or hints to turn off/on hits.")
        move = input().lower()
        if move == 'quit':
            return 'quit'
        if move == 'hints':
            return 'hints'
        if len(move) == 2 and move[0] in DIGITS1TO8 and move[1] in DIGITS1TO8: #입력이 두글자, 1~8의 숫자일 경우
            x = int(move[0]) - 1 # 인덱스는 0부터 시작하므로 1을 뺸다.
            y = int(move[1]) - 1
            if not isValidMove(board, playerTile, x, y): # 유효한 움직임이 아닐 경우 while문 계속 돌린다.
                continue
            else:
                break

        else: #유효한 숫자가 아니다.
            print('This is not a valid move. Type the x digit(1-8), then the y digit (1-8)')
            print('For example, 81 will be the top-right corner.')

    return [x,y] #  리스트를 반환한다.

def getComputerMove(board, computerTile):
    #인공지능 구현
    possibleMoves = getValidMoves(board, computerTile)  # 컴퓨터가 유효한 움직임을 리스트로 저장한다.

    #randomize the order of the possible moves
    random.shuffle(possibleMoves)

    # 1. always go for a corner if available. 코너에 놓는 것이 가장 좋다.
    for x, y in possibleMoves:
        if isOnCorner(x, y):
            return [x, y]

    # 2. Go through all the possible moves and remember the best scoring move.
    # 가장 높은 점수를 받을 수 있는 위치에 대한 리스트를 생선한다.
    bestScore = -1
    for x, y in possibleMoves:  # 가능한 움직임 중에서
        dupeBoard = getBoardCopy(board)
        makeMove(dupeBoard, computerTile, x, y)
        score = getScoreOfBoard(dupeBoard)[computerTile]
        if score > bestScore:  # 최고 점수 갱신
            bestMove = [x,y]
            bestScore = score
    return bestMove

def showPoints(playerTile, computerTile):
    scores = getScoreOfBoard(mainBoard)
    print("You have %s points. The computer has %s points." % (scores[playerTile], scores[computerTile]))


print('Welcome to Othello!')

while True:
    # Reset the board the game.
    mainBoard = getNewBoard()
    resetBoard(mainBoard)
    playerTile, computerTile = enterPlayerTile()
    showHints = False
    turn = whoGoesFirst()
    print('The' + turn + 'will go first.')

    while True:
        if turn == 'player':
            # Player's turn
            if showHints:
                validMovesBoard = getBoardWithValidMoves(mainBoard, playerTile)
                drawBoard(validMovesBoard)
            else:
                drawBoard(mainBoard)
            showPoints(playerTile, computerTile)
            move = getPlayerMove(mainBoard, playerTile)
            if move == 'quit':
                print("Thanks for playing!")
                sys.exit() # terminate the program
            elif move == 'hints':
                showHints = not showHints
                continue
            else:
                makeMove(mainBoard, playerTile, move[0], move[1])

            if getValidMoves(mainBoard, computerTile) == []:
                break
            else:
                turn = 'computer'

        else:
            # Computer's turn
            drawBoard(mainBoard)
            showPoints(playerTile, computerTile)
            input("Press Enter to see the computer\'s move.")
            x, y = getComputerMove(mainBoard, computerTile)
            makeMove(mainBoard, computerTile, x, y)

            if getValidMoves(mainBoard, playerTile) == []:
                break
            else:
                turn = 'player'


    # Display the final score.
    drawBoard(mainBoard)
    scores = getScoreOfBoard(mainBoard)
    print('X scored %s points. O scored %s points.' % (scores['X'], scores['O']))
    if scores[playerTile] > scores[computerTile]:
        print('You beat the computer by %s points!' % (scores[playerTile]- scores[computerTile]))
    elif scores[playerTile] < scores[computerTile]:
        print('You lost. The computer beat you by %s points.' % (scores[computerTile] - scores[playerTile]))
    else:
        print("The game was a tie!")

    if not playAgain():
        break
