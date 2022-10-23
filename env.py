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
        self.jsp = None
        self.observation_space = ObservationSpace()
        self.action_space = ActionSpace()
        self.reset()

    def reset(self):
        self.jsp = JobShopProblem(self.param)
        self.action_space.reset(self.jsp.actions)
        self.observation_space.reset(self.jsp.features())

        return self.observation_space.state

    def step(self, action):
        # Obtain action transition and scheduling it to env
        self.jsp.scheduling(self.action_space.actions[action]())
        # Obtain next state
        next_state = self.jsp.features()
        # Calculate reward
        reward = self.jsp.reward(self.observation_space.state[6],
                                 self.observation_space.state[5],
                                 next_state[6],
                                 next_state[5],
                                 self.observation_space.state[0],
                                 next_state[0])
        # Set env observation state
        self.observation_space.state = next_state

        return next_state, reward, self.jsp.done
