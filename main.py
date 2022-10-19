import numpy as np
import torch
from torch import optim

from dqn import DQN, ReplayMemory
from env import Env
from jsp_utility import Param

# Hyper-parameters
gamma = 0.99
lr = 0.001
replay_memory_capacity = 1000
epochs = 3000
initial_exploration = 1000
log_interval = 1
batch_size = 32
update_target = 100
hidden_layer = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


def train():
    env = Env(Param())
    play_env = Env(Param())
    torch.manual_seed(500)

    # define the state_dim and action_dim
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    online_net = DQN(state_dim, hidden_layer, action_dim)
    target_net = DQN(state_dim, hidden_layer, action_dim)
    target_net.load_state_dict(online_net.state_dict())

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    memory = ReplayMemory(replay_memory_capacity)

    epsilon = 1.0
    steps = 0

    for epoch in range(epochs):
        done = False
        state = env.reset()
        state = torch.FloatTensor(state)

        while not done:
            steps += 1

            # epsilon-greedy policy
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action = target_net.get_action(state)

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state)
            action_one_hot = torch.zeros(action_dim)
            action_one_hot[action] = 1
            mask = 0 if done else 1
            memory.push(state, next_state, action_one_hot, reward, mask)
            state = next_state

            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)

                DQN.train_model(online_net, target_net, optimizer, batch, gamma)

                if steps % update_target == 0:
                    target_net.load_state_dict(online_net.state_dict())

        # Calculate loss function
        play_env.reset(is_play=True)
        total_tardiness = play(play_env, online_net)

        if epoch % log_interval == 0:
            print('{} episode | epsilon: {:.2f} | total_tardiness: {:.2f}'.format(
                epoch, epsilon, total_tardiness))

    torch.save(online_net.state_dict(), 'dqn.pt')


def play(env, online_net):
    state = env.observation_space.state
    state = torch.FloatTensor(state)
    done = False
    while not done:
        action = online_net.get_action(state)
        state, _, done = env.step(action)
        state = torch.FloatTensor(state)

    return sum([max(job.CTK - job.D, 0) for job in env.jsp.jobs])


if __name__ == "__main__":
    train()
