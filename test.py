import numpy as np
import pandas as pd
import torch

from dqn import DQN
from env import Env
from utility import device, dqn_param, jsp_param


def get_tardiness(env):
    return sum([max(job.CTK - job.D, 0) for job in env.jsp.jobs])


def play(env, online_net):
    state = env.observation_space.state
    state = torch.FloatTensor(state)
    done = False
    while not done:
        action = online_net.get_action(state)
        state, _, done = env.step(action)
        state = torch.FloatTensor(state)

    return get_tardiness(env)


def test(sample_num=1):
    tmp_jsp_param = "./tmp_result/jsp.json"
    tmp_net_param = "./tmp_result/dqn.pt"
    env = Env(jsp_param)
    env.jsp.save_jsp_param(tmp_jsp_param)

    # define the state_dim and action_dim
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    online_net = DQN(state_dim, dqn_param.hidden_layer, action_dim)
    online_net.load_state_dict(torch.load(tmp_net_param))
    online_net = online_net.to(device)
    online_net.eval()

    test_result = pd.DataFrame(columns=["online_net", "random"]+[f"rule{i+1}" for i in range(action_dim)])
    count = 0
    for i in range(sample_num):
        round_result = []

        # generate case
        env = Env(jsp_param)
        env.jsp.save_jsp_param(tmp_jsp_param)

        # dqn tardiness
        env = Env(tmp_jsp_param)
        round_result.append(play(env, online_net))

        # random tardiness
        env = Env(tmp_jsp_param)
        env.jsp.play_with_random_rule()
        round_result.append(get_tardiness(env))

        for j in range(action_dim):
            env = Env(tmp_jsp_param)
            env.jsp.play_with_single_rule(j)
            round_result.append(get_tardiness(env))

        if np.argmin(round_result) == 0:
            count += 1

        print(f"round: {i} | count: {count} | tardiness: {round_result[1:]}")
        test_result.loc[len(test_result.index)] = round_result

    return test_result


if __name__ == "__main__":
    test_result = test(1000)
    test_result.to_csv("./tmp_result/test.csv")
