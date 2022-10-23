import random
from collections import namedtuple, deque
import torch
from torch import nn
import torch.nn.functional as F

from utility import device

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward', 'mask'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, state, next_state, action, reward, mask):
        """
        push transition to queue
        """
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self, batch_size):
        """
        sample given number of transitions
        """
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, inputs_dim, hidden_dim, outputs_dim):
        """
        :param inputs_dim: the dimension of inputs
        :param outputs_dim: the dimension of outputs
        """
        super(DQN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inputs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, outputs_dim),
        )

        # initialization of networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        """
        :param: x, input (batch_size, inputs_dim)
        :return: (batch_size, outputs_dim)
        """
        return self.mlp(x)

    def get_action(self, input):
        """
        :param: input (inputs_dim)
        :return:
        """
        input = input.to(device)
        qvalue = self.forward(input)
        _, action = torch.max(qvalue, 0)  # greedy
        return action.cpu().item() if torch.cuda.is_available() else action.item()

    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        states = torch.stack(batch.state).to(device)   # (batch_size, states_dim)
        next_states = torch.stack(batch.next_state).to(device)    # (batch_size, states_dim)
        actions = torch.stack(batch.action).to(device)      # (batch_size, actions_dim)
        rewards = torch.Tensor(batch.reward).to(device)     # (batch_size, )
        masks = torch.Tensor(batch.mask).to(device)         # (batch_size, )

        # pred: (batch_size, action_dim)
        pred = online_net(states)
        _, action_from_online_net = online_net(next_states).max(1)
        next_pred = target_net(next_states)

        # pred: (batch_size, )
        pred = torch.sum(pred.mul(actions), dim=1)

        # pred: (batch_size, )
        target = rewards + masks * gamma * next_pred.gather(1, action_from_online_net.unsqueeze(1)).max(1)[0]

        # L = (rewards + gamma * Q(s', argmaxQ(s', a', w), w-) - Q(s', a', w))^2
        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, rewards.mean()