import numpy as np
import pandas as pd
import torch
from torch import optim

from dqn import DQN, ReplayMemory
from env import Env
from utility import device, dqn_param, jsp_param, MAXINT


def train():
    steps_info = pd.DataFrame(columns=['epochs', 'steps', 'epsilon', 'loss', 'reward'])

    env = Env(jsp_param)
    torch.manual_seed(500)

    # define the state_dim and action_dim
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    online_net = DQN(state_dim, dqn_param.hidden_layer, action_dim)
    target_net = DQN(state_dim, dqn_param.hidden_layer, action_dim)
    target_net.load_state_dict(online_net.state_dict())

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()

    optimizer = optim.Adam(online_net.parameters(), lr=dqn_param.lr)

    memory = ReplayMemory(dqn_param.replay_memory_capacity)

    epsilon = 0.5
    steps = 0
    loss = MAXINT
    should_finished_early = False

    for epoch in range(dqn_param.epochs):
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

            if steps > dqn_param.initial_exploration:
                epsilon -= 0.0005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(dqn_param.batch_size)
                loss, reward = DQN.train_model(online_net, target_net, optimizer, batch, dqn_param.gamma)

                # Update target network parameters.
                for target_param, param in zip(target_net.parameters(), online_net.parameters()):
                    target_param.data.copy_(dqn_param.tau * param + (1 - dqn_param.tau) * target_param)

                steps_info.loc[len(steps_info.index)] = [epoch, steps, epsilon,
                                                         loss.detach().item(),
                                                         reward.detach().item()]
                if loss < 0.001 or steps > 10000:
                    should_finished_early = True
                    break

        if epoch % dqn_param.log_interval == 0:
            print(f"{epoch} episode | epsilon: {epsilon:.2f} | steps: {steps} | loss: {loss:.2f} | reward: {reward:.2f}")

        if should_finished_early:
            break

    steps_info.to_csv("./tmp_result/steps_info.csv", index=False)
    torch.save(online_net.state_dict(), './tmp_result/dqn.pt')


if __name__ == "__main__":
    train()
