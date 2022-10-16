import random

from jsp import JobShopProblem


class ObservationSpace(object):
    def __init__(self):
        self.state = None
        self.shape = None

    def reset(self, state):
        """
        :param state: np.array
        :return:
        """
        self.state = state
        self.shape = self.state.shape


class ActionSpace(object):
    def __init__(self):
        self.actions = None
        self.n = None

    def reset(self, actions):
        """
        :param actions: list[fun]
        :return:
        """
        self.actions = actions
        self.n = len(self.actions)

    def sample(self):
        return random.randint(0, self.n-1)


class Env(object):
    def __init__(self, param):
        self.param = param
        self.env = None
        self.backup_env = None
        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace()
        self.reset()

    def reset(self, is_play=False):
        if not self.backup_env:
            self.env = JobShopProblem(self.param)
            self.backup_env = self.env
        elif is_play:
            self.env = self.backup_env
        else:
            self.env = JobShopProblem(self.param)

        self.action_space.reset(self.env.actions)
        self.observation_space.reset(self.env.features())

        return self.observation_space.state

    def step(self, action):
        # Obtain action transition and scheduling it to env
        self.env.scheduling(self.action_space.actions[action]())
        # Obtain next state
        next_state = self.env.features()
        # Calculate reward
        reward = self.env.reward(self.observation_space.state[6],
                                 self.observation_space.state[5],
                                 next_state[6],
                                 next_state[5],
                                 self.observation_space.state[0],
                                 next_state[0])
        # Set env observation state
        self.observation_space.state = next_state

        return next_state, reward, self.env.done
