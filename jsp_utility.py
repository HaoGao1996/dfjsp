import random


class Param(object):
    def __init__(self,
                 M_num=5,
                 J_init_num=10,
                 J_insert_num=20,
                 DDT=0.5,
                 max_op_num=10,
                 max_processing_time=20,
                 pb_jop=0.3,
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


class Operation(object):
    def __init__(self, idx, param):
        self.idx = idx
        self.t_ijk = [-1 if random.random() < param.pb_jop
                      else random.randint(1, param.max_processing_time)
                      for _ in range(param.M_num)]
        if self.t_ijk.count(-1) == param.M_num:
            self.t_ijk[random.randint(0, param.M_num - 1)] = \
                random.randint(1, param.max_processing_time)
        self.t_ij_ave = (sum(self.t_ijk) + self.t_ijk.count(-1)) / (param.M_num - self.t_ijk.count(-1))


class Job(object):
    def __init__(self, idx, param):
        self.idx = idx
        self.op_num = random.randint(1, param.max_op_num)  # Operation number
        self.operations = [Operation(i, param) for i in range(self.op_num)]  # Set of operations
        self.A = 0 if self.idx < param.J_init_num \
            else int(random.expovariate(1 / param.E_ave))  # Arrival time of job
        self.t_ij = [op.t_ij_ave for op in self.operations]
        self.D = int(sum(self.t_ij) * param.DDT) if self.idx < param.J_init_num \
            else self.A + int(sum(self.t_ij) * param.DDT)  # Due date of a job

        self.CTK = 0  # The completion time of the last operation of job
        self.OP = 0  # The number of completed operations of job
        self.CRJ = 0  # The completion rate of job
        self.is_uncompleted = True  # Checks if OP < op_num
        self.is_tardiness = False  # Checks if OP < op_num and CTK > D
        self.estimated_t_ij = sum(self.t_ij[self.OP:])

    def update(self, end_time):
        self.CTK = end_time
        self.OP += 1
        self.CRJ = self.OP / self.op_num
        self.is_uncompleted = self.OP < self.op_num
        self.is_tardiness = self.is_uncompleted and self.CTK > self.D
        self.estimated_t_ij = sum(self.t_ij[self.OP:]) if self.is_uncompleted else 0


class Machine(object):
    def __init__(self, idx):
        self.idx = idx
        self.CTK = 0  # The completion time of the last operation of machine
        self.ITK = 0  # The accumulated idle time of machine
        self.UK = 0  # The utilization rate of machine

    def update(self, start_time, end_time):
        self.ITK += start_time - self.CTK
        self.CTK = end_time
        self.UK = 1 - self.ITK / self.CTK
