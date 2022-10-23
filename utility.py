import sys
import torch


class DQNParam(object):
    def __init__(self,
                 gamma=0.9,
                 lr=0.001,
                 replay_memory_capacity=500,
                 epochs=50,
                 initial_exploration=1000,
                 log_interval=1,
                 batch_size=32,
                 update_target=100,
                 hidden_layer=32,
                 tau=0.01):
        self.gamma = gamma
        self.lr = lr
        self.replay_memory_capacity = replay_memory_capacity
        self.epochs = epochs
        self.initial_exploration = initial_exploration
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.update_target = update_target
        self.hidden_layer = hidden_layer
        self.tau = tau


class JSPParam(object):
    def __init__(self,
                 M_num=20,
                 J_init_num=20,
                 J_insert_num=50,
                 DDT=0.5,
                 max_op_num=20,
                 max_processing_time=50,
                 pb_jop=0.5,
                 E_ave=50):
        """
        :param M_num: machine number
        :param J_init_num: job initial number
        :param J_insert_num: job insert number
        :param DDT: due date tightness
        :param max_op_num: maximum operation number of job
        :param max_processing_time: maximum processing time of operation
        :param pb_jop: matching probability of job, operation, machine
        :param E_ave: average value of exponential distribution Between two successive new job arrivals
        """
        self.M_num = M_num
        self.J_init_num = J_init_num
        self.J_insert_num = J_insert_num
        self.J_num = J_init_num + J_insert_num
        self.DDT = DDT
        self.max_op_num = max_op_num
        self.max_processing_time = max_processing_time
        self.pb_jop = pb_jop
        self.E_ave = E_ave


MAXINT = sys.maxsize
MININT = -sys.maxsize - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device =  torch.device('cpu')
dqn_param = DQNParam()
jsp_param = JSPParam()
