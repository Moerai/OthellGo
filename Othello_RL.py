import os, copy
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from collections import deque
from IPython.display import clear_output
import matplotlib.pyplot as plt

import cnndqn
import replaybuffer

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

model = cnndqn.CnnDQN(8*8, 8*8)

optimizer = optim.Adam(model.parameters(), lr=0.00001)

replay_initial = 10000
replay_buffer = replaybuffer.ReplayBuffer(100000)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

n = 8  # board size (even)

board = [['0' for x in range(n)] for y in range(n)]
# 8 directions
dirx = [-1, 0, 1, -1, 1, -1, 0, 1]
diry = [-1, -1, -1, 0, 0, 1, 1, 1]

opt = 2
depth = 4

num_frames = 14000
batch_size = 32
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


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
                tot += 1
    return tot


def IsTerminalNode(board, player):
    for y in range(n):
        for x in range(n):
            if ValidMove(board, x, y, player):
                return False
    return True


def BestMove(board, player):
    maxPoints = 0
    mx = -1; my = -1
    for y in range(n):
        for x in range(n):
            if ValidMove(board, x, y, player):
                (boardTemp, totctr) = MakeMove(copy.deepcopy(board), x, y, player)
                points = AlphaBeta(board, player, depth, minEvalBoard, maxEvalBoard, True)
                if points > maxPoints:
                    maxPoints = points
                    mx = x; my = y
    return (mx, my)

def AlphaBeta(board, player, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or IsTerminalNode(board, player):
        return EvalBoard(board, player)
    if maximizingPlayer:
        v = minEvalBoard
        for y in range(n):
            for x in range(n):
                if ValidMove(board, x, y, player):
                    (boardTemp, totctr) = MakeMove(copy.deepcopy(board), x, y, player)
                    v = max(v, AlphaBeta(boardTemp, player, depth - 1, alpha, beta, False))
                    alpha = max(alpha, v)
                    if beta <= alpha:
                        break # beta cut-off
        return v
    else: # minimizingPlayer
        v = maxEvalBoard
        for y in range(n):
            for x in range(n):
                if ValidMove(board, x, y, player):
                    (boardTemp, totctr) = MakeMove(copy.deepcopy(board), x, y, player)
                    v = min(v, AlphaBeta(boardTemp, player, depth - 1, alpha, beta, True))
                    beta = min(beta, v)
                    if beta <= alpha:
                        break # alpha cut-off
        return v


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


def get_validlist(board, player):
    validlist = []
    for x in range(8):
        for y in range(8):
            if ValidMove(board, x, y, player):
                validlist.append([x, y])
    return validlist


def random_agent(board, player):
    validlist = get_validlist(board, player)
    random.shuffle(validlist)
    (x, y) = validlist.pop()
    x = int(x)
    y = int(y)
    (board, totctr) = MakeMove(board, x, y, player)
    print('player'+player+ 'played (X Y): ' + str(x) + ' ' + str(y))
    print('# of pieces taken: ' + str(totctr))
    PrintBoard()

def alphabeta_agent(board, player):
    (x, y) = BestMove(board, player)
    if not (x == -1 and y == -1):
        (board, totctr) = MakeMove(board, x, y, player)
        print('AlphaBeta played (X Y): ' + str(x) + ' ' + str(y))
        print('# of pieces taken: ' + str(totctr))


state = board
num_episode = 100
is_terminal = False

for i in range(num_episode):

    # for frame_idx in range(1, num_frames+1):
    state = [['0' for x in range(n)] for y in range(n)]

    InitBoard()
    player_terminal = [0, 0]
    is_terminal = False
    # 한 게임
    while not is_terminal:
        for p in range(2):
            print
            # PrintBoard()
            player = str(p + 1)
            print('PLAYER: ' + player)
            if IsTerminalNode(board, player):
                player_terminal[p] = 1
                if 0 in player_terminal:
                    continue
                else:
                    is_terminal = True
                    print('Player cannot play! Game ended!')
                    print('Score Comp1: ' + str(EvalBoard(board, '1')))
                    print('Score Comp2: ' + str(EvalBoard(board, '2')))
                    # 이겼으면
                    if EvalBoard(board, '2') > EvalBoard(board, '1'):
                        reward = 10
                    elif EvalBoard(board, '2') == EvalBoard(board, '1'):
                        reward = 0
                    else:
                        reward = -5

                    # discounted sum of reward 해서 각 에피소드별 reward 구해야할듯
                    # 10판마다 학습시키는건 어디서하는거지?
                    all_rewards.append(episode_reward)
                    episode_reward = 0

                    if len(replay_buffer) > replay_initial:
                        loss = compute_td_loss(batch_size)
                        losses.append(loss.data[0])

                    # if frame_idx % 10000 == 0:
                    #     plot(frame_idx, all_rewards, losses)


                # 한 게임 끝
                break
                # os._exit(0)

            if player == '1': #computer1's turn
                random_agent(board, player)

            else:  # computer2's turn (학습시킬아이)
                epsilon = epsilon_by_frame(frame_idx)
                validlist = get_validlist(board, player)
                action = model.act(state, epsilon, validlist)  # return (x,y)
                x = action[0]
                y = action[1]
                state = board
                (board, totctr) = MakeMove(board, x, y, player)
                next_state, reward, done = board, 0, False
                replay_buffer.push(state, action, reward, next_state, done)

                print('player' + player + 'played (X Y): ' + str(x) + ' ' + str(y))
                print('# of pieces taken: ' + str(totctr))
                PrintBoard()