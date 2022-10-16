import random
import numpy as np


class Param(object):
    def __init__(self,
                 M_num = 10,
                 J_init_num = 20,
                 J_insert_num = 50,
                 DDT = 0.5,
                 max_op_num = 20,
                 max_processing_time = 50,
                 pb_jop = 0.3,
                 E_ave = 50):
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


class Generator(object):
    def __init__(self, param):
        """
        Generate learning instance.
        """
        self.M_num = param.M_num
        self.J_init_num = param.J_init_num
        self.J_insert_num = param.J_insert_num
        self.J_num = self.J_init_num + self.J_insert_num
        self.DDT = param.DDT
        self.max_op_num = param.max_op_num
        self.max_processing_time = param.max_processing_time
        self.pb_jop = param.pb_jop
        self.E_ave = param.E_ave
        self._generate()

    def _generate(self):
        # Generate operation num matrix (1 * J_num)
        self.Op_num = [random.randint(1, self.max_op_num) for _ in range(self.J_num)]
        self.O_num = sum(self.Op_num)
        self.J = dict(enumerate(self.Op_num))
        # Generate processing time matrix (J_num * Op_num[i] * M_num)
        self.processing_time = [[[-1 if random.random() < self.pb_jop else random.randint(1, self.max_processing_time)
                                for _ in range(self.M_num)] for _ in range(self.Op_num[i])]
                                for i in range(self.J_num)]
        # Generate the arrival time of jobs (1 * J_num)
        self.A = [0] * self.J_init_num + np.random.exponential(self.E_ave, size=self.J_insert_num).astype(int).tolist()
        # Generate the due date of jobs (1 * J_num)
        def get_processing_time_ij_ave(t_ij):
            T_ijk = [k for k in t_ij if k != -1]
            return sum(T_ijk) / len(T_ijk)
        T_ij_ave = [get_processing_time_ij_ave(self.processing_time[i][j]) for i in range(self.J_num) for j in range(self.Op_num[i])]
        self.D = [int(T_ij_ave[i] * self.DDT) for i in range(self.J_init_num)] + \
                 [self.A[i] + int(T_ij_ave[i] * self.DDT) for i in range(self.J_init_num, self.J_num)]

    def print_param(self):
        print(f"M_num: {self.M_num}, DDT: {self.DDT}, "
              f"max_op_num: {self.max_op_num}, max_processing_time: {self.max_processing_time}, "
              f"pb_jop: {self.pb_jop}, E_ave: {self.E_ave}")
        print(f"J_num: {self.J_num}, J_init_num: {self.J_init_num}, J_insert_num: {self.J_insert_num}, O_num: {self.O_num}")
        print(f"J: {self.J}")
        print(f"Op_num: {self.Op_num}")
        print(f"A: {self.A}")
        print(f"D: {self.D}")
        print(f"processing_time: {self.processing_time}")


if __name__ == "__main__":
    param = Param()
    instance = Generator(param)
    instance.print_param()