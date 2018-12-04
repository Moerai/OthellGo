import os, copy
import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from collections import deque
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape  # 8*8 state(observation) size
        self.num_actions = num_actions  # 8*8 action size 난 똑같

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            # output는 6*6
            nn.ReLU(),
        )

        # fully connected = 그냥 신경망
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon, validlist):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)  # size 64의 linear [1,2,3,4]
            # q_value에서 valid 한 것만 놔두기.
            valid_q_value = []
            for i in validlist:
                idx = i[0]*8 + i[1]
                valid_q_value.append(q_value[idx])
            # action = valid_q_value.max(1)[1].data[0]
            val = np.max(valid_q_value)
            val_idx = (q_value == val).nonzero.item()
            # 해당 인덱스의 좌표 찾기
            x = int(val_idx//8)
            y = int(val_idx % 8)
            action = (x, y)

        else:
            random.shuffle(validlist)
            (x, y) = validlist.pop()
            x = int(x)
            y = int(y)
            action = (x, y)

        return action
