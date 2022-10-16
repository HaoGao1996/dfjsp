import numpy as np
import random


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


class Object(object):
    def __init__(self, I):
        self.I = I
        self.Start = []
        self.End = []
        self.T = []
        self.assign_for = []

    def _add(self, S, E, obs, t):
        # obs:安排的对象
        self.Start.append(S)
        self.End.append(E)
        self.Start.sort() #默认升序排列
        self.End.sort()
        self.T.append(t)
        self.assign_for.insert(self.End.index(E), obs)

    def idle_time(self): #闲置时间
        Idle=[]
        try:
            if self.Start[0]!=0:
                Idle.append([0,self.Start[0]])
            K=[[self.End[i],self.Start[i+1]] for i in range(len(self.End)) if self.Start[i+1]-self.End[i]>0]#如果下个工件的开始时间大于上个工件的结束时间
            Idle.extend(K)
        except:
            pass
        return  Idle


class JobShopProblem(object):
    def __init__(self, param):
        self.param = param
        self.M_num = param.M_num  # 机器数
        self.J_num = param.J_init_num + param.J_insert_num  # 工件数
        self.CTK = [0] * self.M_num  # 各机器上最后一道工序的完工时间列表
        self.OP = [0] * self.J_num  # 各工件的已加工工序数列表
        self.UK = [0] * self.M_num  # 各机器的实际使用率
        self.CRJ = [0] * self.J_num  # 工件完工率
        self.Jobs = [Object(i) for i in range(self.J_num)]  # 工件集
        self.Machines = [Object(i) for i in range(self.M_num)]  # 机器集
        self._generate()
        self.actions = [self.rule1, self.rule2, self.rule3, self.rule4, self.rule5, self.rule5]  # rules
        self.finished_O_num = 0
        self.done = False

    def _generate(self):
        # Generate operation num matrix (1 * J_num)
        self.Op_num = [random.randint(1, self.param.max_op_num) for _ in range(self.J_num)]
        self.O_num = sum(self.Op_num)
        self.J = dict(enumerate(self.Op_num))
        # Generate processing time matrix (J_num * Op_num[i] * M_num)
        self.Processing_time = []
        for i in range(self.J_num):
            job_processing_time = []
            for _ in range(self.Op_num[i]):
                op_processing_time = [-1] * self.M_num
                for j in range(self.M_num):
                    if random.random() < self.param.pb_jop:
                        op_processing_time[j] = random.randint(1, self.param.max_processing_time)
                # if all value are -1
                if op_processing_time.count(-1) == self.M_num:
                    op_processing_time[random.randint(0, self.M_num-1)] = \
                        random.randint(1, self.param.max_processing_time)
                job_processing_time.append(op_processing_time)
            self.Processing_time.append(job_processing_time)

        # Generate the arrival time of jobs (1 * J_num)
        self.A = [0] * self.param.J_init_num + np.random.exponential(self.param.E_ave, size=self.param.J_insert_num).astype(int).tolist()

        # Generate the due date of jobs (1 * J_num)
        def get_processing_time_ij_ave(t_ij):
            T_ijk = [k for k in t_ij if k != -1]
            return sum(T_ijk) / len(T_ijk)

        T_ij_ave = [get_processing_time_ij_ave(self.Processing_time[i][j]) for i in range(self.J_num) for j in
                    range(self.Op_num[i])]
        self.D = [int(T_ij_ave[i] * self.param.DDT) for i in range(self.param.J_init_num)] + \
                 [self.A[i] + int(T_ij_ave[i] * self.param.DDT) for i in range(self.param.J_init_num, self.J_num)]

    def _update(self, job, machine):
        self.CTK[machine] = max(self.Machines[machine].End)
        self.OP[job] += 1
        try:
            self.UK[machine] = sum(self.Machines[machine].T) / self.CTK[machine]
        except ZeroDivisionError:
            self.UK[machine] = 0
        self.CRJ[job] = self.OP[job] / self.J[job]

    # 机器平均使用率
    def features(self):
        # 1 机器平均利用率
        U_ave = sum(self.UK) / self.M_num
        K = 0
        for uk in self.UK:
            K += np.square(uk - U_ave)
        # 2 机器的使用率标准差
        U_std = np.sqrt(K / self.M_num)
        # 3 平均工序完成率
        CRO_ave = sum(self.OP) / self.O_num
        # 4 平均工件工序完成率
        CRJ_ave = sum(self.CRJ) / self.J_num
        K = 0
        for uk in self.CRJ:
            K += np.square(uk - CRJ_ave)
        # 5 工件工序完成率标准差
        CRJ_std = np.sqrt(K / self.J_num)
        # 6 Estimated tardiness rate Tard_e
        T_cur = sum(self.CTK) / self.M_num
        N_tard, N_left = 0, 0
        for i in range(self.J_num):
            if self.J[i] > self.OP[i]:
                N_left += self.J[i] - self.OP[i]
                T_left = 0
                for j in range(self.OP[i], self.J[i]):
                    M_ij = [k for k in self.Processing_time[i][j] if k > 0 or k < 999]
                    T_left += sum(M_ij) / len(M_ij)
                    if T_left + T_cur > self.D[i]:
                        N_tard += self.J[i] - j+1
                        break
        try:
            Tard_e = N_tard / N_left
        except:
            Tard_e = 9999
        # 7 Actual tardiness rate Tard_a
        N_tard, N_left = 0, 0
        for i in range(self.J_num):
            if self.J[i] > self.OP[i]:
                N_left += self.J[i] - self.OP[i]
                try:
                    if max(self.Jobs[i].End) > self.D[i]:
                        N_tard += self.J[i] - self.OP[i]
                except:
                    pass
        try:
            Tard_a = N_tard / N_left
        except:
            Tard_a = 9999
        return np.array([U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, Tard_e, Tard_a])

    # Composite dispatching rule 1
    # return Job,Machine
    def rule1(self):
        # T_cur:平均完工时间
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:不能按期完成的工件
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur] #现在工件没被加工完并且平均完工时间比此工件的交货时间晚
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]] #此时没被加工完的工件
        if Tard_Job == []: #如果不存在不能按期完成的工件
            Job_i = UC_Job[np.argmin([(self.D[i] - T_cur) / (self.J[i] - self.OP[i]) for i in UC_Job])] #先加工离交货期最近的工件
        else: #如果存在不能按期完成的工件
            T_ijave = []
            for i in Tard_Job:
                Tad = []
                for j in range(self.OP[i], self.J[i]): #此工件剩下的工序中
                    T_ijk = [k for k in self.Processing_time[i][j] ]
                    Tad.append(sum(T_ijk) / len(T_ijk))#某一道工序的平均加工时间 sum(Tad)为该工件的未加工完的工序的平均加工时间
                T_ijave.append(T_cur + sum(Tad) - self.D[i])
            Job_i = Tard_Job[np.argmax(T_ijave)] #从待加工工件中选工件
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.A[Job_i]  # 工件i的arrival time
        A_ij = self.A[Job_i]  # 工件i的arrival time
        # print(A_ij)

        Mk = []
        for i in range(len(self.CTK)):
            Mk.append(max(C_ij, A_ij, self.CTK[i]))

        # print('This is from rule 1:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 1:',Machine)
        return Job_i, Machine

    def rule2(self):
            # T_cur:平均完工时间
            T_cur = sum(self.CTK) / self.M_num
            # Tard_Job:不能按期完成的工件
            Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
            T_ijave = []
            for i in range(self.J_num):
                Tad = []
                for j in range(self.OP[i], self.J[i]):
                    T_ijk = [k for k in self.Processing_time[i][j] ]
                    Tad.append(sum(T_ijk) / len(T_ijk))
                T_ijave.append(sum(Tad))
            if Tard_Job == []:
                Job_i = UC_Job[np.argmin([(self.D[i] - T_cur) / T_ijave[i] for i in UC_Job])]
            else:
                Job_i = Tard_Job[np.argmax([T_cur + T_ijave[i] - self.D[i] for i in Tard_Job])]
            try:
                C_ij = max(self.Jobs[Job_i].End)
            except:
                C_ij = self.A[Job_i]  # 工件i的arrival time
            A_ij = self.A[Job_i]  # 工件i的arrival time
            # print(A_ij)

            Mk = []
            for i in range(len(self.CTK)):

                Mk.append(max(C_ij, A_ij, self.CTK[i]))

            # print('This is from rule 2:',Mk)
            Machine = np.argmin(Mk)
            # print('This is from rule 2:',Machine)
            return Job_i, Machine


    def rule3(self):
        # T_cur:平均完工时间
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:不能按期完成的工件
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        T_ijave = []
        for i in UC_Job:
            Tad = []
            for j in range(self.OP[i], self.J[i]):
                T_ijk = [k for k in self.Processing_time[i][j]]
                Tad.append(sum(T_ijk) / len(T_ijk))
            T_ijave.append(T_cur + sum(Tad) - self.D[i])
        Job_i = UC_Job[np.argmax(T_ijave)]
        On = len(self.Jobs[Job_i].End)
        if random.random() < 0.5:
            U = []
            for i in range(len(self.UK)):

                    U.append(self.UK[i])
            Machine = np.argmin(U)
        else:
            MT = []
            for j in range(self.M_num):
               MT.append(sum(self.Machines[j].T))
            Machine = np.argmin(MT)
        # print('This is from rule 3:',Machine)
        return Job_i, Machine


    def rule4(self):
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
            Job_i = random.choice(UC_Job)
            try:
                C_ij = max(self.Jobs[Job_i].End)
            except:
                C_ij = self.A[Job_i]  # 工件i的arrival time
            A_ij = self.A[Job_i]  # 工件i的arrival time

            Mk = []
            for i in range(len(self.CTK)):
                Mk.append(max(C_ij, A_ij, self.CTK[i]))
            # print('This is from rule 4:',Mk)
            Machine = np.argmin(Mk)
            # print('This is from rule 4:',Machine)
            return Job_i, Machine


    def rule5(self):
        # T_cur:平均完工时间
        T_cur = sum(self.CTK) / self.M_num
        # Tard_Job:不能按期完成的工件
        Tard_Job = [i for i in range(self.J_num) if self.OP[i] < self.J[i] and self.D[i] < T_cur]
        UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
        if Tard_Job == []:
            Job_i = UC_Job[np.argmin([self.CRJ[i] * (self.D[i] - T_cur) for i in UC_Job])]
        else:
            T_ijave = []
            for i in Tard_Job:
                Tad = []
                for j in range(self.OP[i], self.J[i]):
                    T_ijk = [k for k in self.Processing_time[i][j] ]
                    Tad.append(sum(T_ijk) / len(T_ijk))
                T_ijave.append(1 / (self.CRJ[i]+1 ) * (T_cur + sum(Tad) - self.D[i]))
            Job_i = Tard_Job[np.argmax(T_ijave)]
        try:
            C_ij = max(self.Jobs[Job_i].End)
        except:
            C_ij = self.A[Job_i]  # 工件i的arrival time
        A_ij = self.A[Job_i]  # 工件i的arrival time

        Mk = []
        for i in range(len(self.CTK)):

            Mk.append(max(C_ij, A_ij, self.CTK[i]))

        # print('This is from rule 5:',Mk)
        Machine = np.argmin(Mk)
        # print('This is from rule 5:',Machine)
        return Job_i, Machine

    def rule6(self):
            # T_cur:平均完工时间
            T_cur = sum(self.CTK) / self.M_num
            UC_Job = [j for j in range(self.J_num) if self.OP[j] < self.J[j]]
            T_ijave = []
            for i in UC_Job:
                Tad = []
                for j in range(self.OP[i], self.J[i]):
                    T_ijk = [k for k in self.Processing_time[i][j] ]
                    Tad.append(sum(T_ijk) / len(T_ijk))
                T_ijave.append(T_cur + sum(Tad) - self.D[i])
            Job_i = UC_Job[np.argmax(T_ijave)]
            try:
                C_ij = max(self.Jobs[Job_i].End)
            except:
                C_ij = self.A[Job_i]  # 工件i的arrival time
            A_ij = self.A[Job_i]  # 工件i的arrival time

            Mk = []
            for i in range(len(self.CTK)):

                Mk.append(max(C_ij, A_ij, self.CTK[i]))

            Machine = np.argmin(Mk)
            # print('this is from rule 6:',Mk)
            # print('This is from rule 6:',Machine)
            return Job_i, Machine


    def scheduling(self, action):
        Job, Machine = action[0], action[1]
        O_n = len(self.Jobs[Job].End) #已经加工完的工序数
        # print(Job, Machine,O_n)

        try:
            last_ot = max(self.Jobs[Job].End)  # 某工件上道工序加工完成时间
        except:
            last_ot = 0
        try:
            last_mt = max(self.Machines[Machine].End)  # 机器上最后一道工序完工时间
        except:
            last_mt = 0
        Start_time = max(last_ot, last_mt) #下一道工序开始加工时间

        PT = self.Processing_time[Job][O_n][Machine]  # 下一道工序在机器machine上的加工时间

        end_time = Start_time + PT
        self.Machines[Machine]._add(Start_time, end_time, Job, PT)
        self.Jobs[Job]._add(Start_time, end_time, Machine, PT)
        self._update(Job, Machine)

        self.finished_O_num += 1
        # If all operations have been scheduled, end game.
        if self.finished_O_num == self.O_num - 1:
            self.done = True


    def reward(self, Ta_t, Te_t, Ta_t1, Te_t1, U_t, U_t1):
        """
        :param Ta_t: Tard_a(t)
        :param Te_t: Tard_e(t)
        :param Ta_t1: Tard_a(t+1)
        :param Te_t1: Tard_e(t+1)
        :param U_t: U_ave(t)
        :param U_t1: U_ave(t+1)
        :return: reward
        """
        if Ta_t1 < Ta_t:
            rt = 1
        else:
            if Ta_t1 > Ta_t:
                rt = -1
            else:
                if Te_t1 < Te_t:
                    rt = 1
                else:
                    if Te_t1 > Te_t:
                        rt = -1
                    else:
                        if U_t1 > U_t:
                            rt = 1
                        else:
                            if U_t1 > 0.95 * U_t:
                                rt = 0
                            else:
                                rt = -1
        return rt

    def print_param(self):
        print(f"M_num: {self.M_num}, DDT: {self.param.DDT}, "
              f"max_op_num: {self.param.max_op_num}, max_processing_time: {self.param.max_processing_time}, "
              f"pb_jop: {self.param.pb_jop}, E_ave: {self.param.E_ave}")
        print(f"J_num: {self.J_num}, J_init_num: {self.param.J_init_num}, J_insert_num: {self.param.J_insert_num}, O_num: {self.O_num}")
        print(f"J: {self.J}")
        print(f"Op_num: {self.Op_num}")
        print(f"A: {self.A}")
        print(f"D: {self.D}")
        print(f"processing_time: {self.Processing_time}")


if __name__ == "__main__":
    jsp = JobShopProblem(Param())
    jsp.print_param()

