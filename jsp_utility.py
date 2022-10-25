import random

from utility import JSPParam


class JSPInputError(Exception):
    def __init__(self, error_info):
        super().__init__(self)
        self.error_info = error_info

    def __str__(self):
        return self.error_info


class Operation(object):
    def __init__(self, idx, param):
        self.idx = idx
        if isinstance(param, JSPParam):
            self.t_ijk = [-1 if random.random() < 0.5
                          else random.randint(1, param.max_processing_time)
                          for _ in range(param.M_num)]
            if self.t_ijk.count(-1) == param.M_num:
                self.t_ijk[random.randint(0, param.M_num - 1)] = \
                    random.randint(1, param.max_processing_time)
            self.t_ij_ave = (sum(self.t_ijk) + self.t_ijk.count(-1)) / (param.M_num - self.t_ijk.count(-1))
        elif isinstance(param, dict):
            self.t_ijk = param["t_ijk"]
            self.t_ij_ave = param["t_ij_ave"]
        else:
            raise JSPInputError("No available param: operation")

    def save_operation_param(self):
        # Data to be written
        dictionary = {
            "t_ijk": self.t_ijk,
            "t_ij_ave": self.t_ij_ave
        }

        return dictionary


class Job(object):
    def __init__(self, idx, param):
        self.idx = idx
        if isinstance(param, JSPParam):
            self.op_num = random.randint(1, param.max_op_num)  # Operation number
            self.operations = [Operation(i, param) for i in range(self.op_num)]  # Set of operations
            self.A = 0 if self.idx < param.J_init_num \
                else int(random.expovariate(1 / param.E_ave)) + 1  # Arrival time of job
            self.t_ij = [op.t_ij_ave for op in self.operations]
            self.D = self.A + int(sum(self.t_ij) * param.DDT)  # Due date of a job
        elif isinstance(param, dict):
            self.op_num = param["op_num"]
            self.operations = [Operation(int(idx), operation) for idx, operation in param["operations"].items()]
            self.A = param["A"]
            self.D = param["D"]
            self.t_ij = [op.t_ij_ave for op in self.operations]
        else:
            raise JSPInputError("No available param: job")

        self.CTK = 0  # The completion time of the last operation of job
        self.OP = 0  # The number of completed operations of job
        self.CRJ = 0  # The completion rate of job
        self.is_uncompleted = True  # Checks if OP < op_num
        self.estimated_t_ij = sum(self.t_ij[self.OP:])

    def update(self, end_time):
        self.CTK = end_time
        self.OP += 1
        self.CRJ = self.OP / self.op_num
        self.is_uncompleted = self.OP < self.op_num
        self.estimated_t_ij = sum(self.t_ij[self.OP:]) if self.is_uncompleted else 0

    def save_job_param(self):
        # Data to be written
        dictionary = {
            "op_num": self.op_num,
            "operations": {operation.idx: operation.save_operation_param() for operation in self.operations},
            "A": self.A,
            "D": self.D
        }

        return dictionary


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
