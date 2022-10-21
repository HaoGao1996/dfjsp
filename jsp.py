import numpy as np
import random
import statistics as s
import sys

from jsp_utility import Machine, Job, Param

MAXINT = sys.maxsize
MININT = -sys.maxsize - 1


class JobShopProblem(object):
    def __init__(self, param):
        self.param = param  # Basic parameters
        self.M_num = param.M_num  # Machines number
        self.machines = [Machine(i) for i in range(self.M_num)]  # Set of machines
        self.J_num = param.J_num  # Jobs' number
        self.jobs = [Job(i, param) for i in range(self.J_num)]  # Set of jobs
        self.actions = [self.rule1, self.rule2, self.rule3, self.rule4, self.rule5, self.rule5]  # Rules
        self.policy_list = []

        self.total_op_num = sum([job.op_num for job in self.jobs])  # Total operation number
        self.finished_O_num = 0  # Finished operation number
        self.done = False  # Flag if jsp is solved

        self.T_cur = 0  # Average completion time of the last operations on all machines
        self.UC_job = [job for job in self.jobs if job.is_uncompleted]  # Uncompleted jobs
        self.Tard_job = [job for job in self.jobs if job.is_tardiness]  # Tardiness jobs

    def scheduling(self, action):
        job, machine = self.jobs[action[0]], self.machines[action[1]]
        operation = job.operations[job.OP]
        self.policy_list.append([job.idx, operation.idx, machine.idx])

        if operation.t_ijk[machine.idx] == -1:
            print("error")

        start_time = max(job.CTK, machine.CTK)  # Next operation start time
        end_time = start_time + operation.t_ijk[machine.idx]

        machine.update(start_time, end_time)
        job.update(end_time)

        self.T_cur = s.mean([machine.CTK for machine in self.machines])
        self.UC_job = [job for job in self.jobs if job.is_uncompleted]
        self.Tard_job = [job for job in self.jobs if job.is_tardiness]

        # If all operations have been scheduled, end game.
        self.finished_O_num += 1
        if self.finished_O_num == self.total_op_num - 1:
            self.done = True

    def _estimated_tardiness_rate(self):
        N_left = sum([job.op_num - job.OP for job in self.UC_job])
        N_tard = 0
        for job in self.UC_job:
            T_left = 0
            for i in range(job.OP, job.op_num):
                T_left += job.operations[i].t_ij_ave
                if T_left + self.T_cur > job.D:
                    N_tard += job.op_num - i + 1
                    break
        return N_tard / N_left

    def _actual_tardiness_rate(self):
        N_left = sum([job.op_num - job.OP for job in self.UC_job])
        N_tard = sum([job.op_num - job.OP for job in self.Tard_job])

        return N_tard / N_left

    def features(self):
        """
        :return:
        Feature 1: The Average utilization rate of machines
        Feature 2: The standard deviation of machine utilization rate
        Feature 3: The average completion rate of operations
        Feature 4: The average completion rate of jobs
        Feature 5:  The standard deviation of job completion rate
        Feature 5:  The standard deviation of job completion rate
        Feature 6: Estimated tardiness rate Tard_e
        Feature 7: Actual tardiness rate Tard_a
        """
        UK_list = [machine.UK for machine in self.machines]
        OP_list = [job.OP for job in self.jobs]
        CRJ_list = [job.CRJ for job in self.jobs]

        return np.array([s.mean(UK_list), s.pstdev(UK_list),
                         s.mean(OP_list) / self.total_op_num,
                         s.mean(CRJ_list), s.pstdev(CRJ_list),
                         self._estimated_tardiness_rate(),
                         self._actual_tardiness_rate()])

    # Composite dispatching rule 1
    def rule1(self):
        # Select job
        if not self.Tard_job:
            job = self.UC_job[
                np.argmin(
                    [(job.D - self.T_cur) / (job.op_num - job.OP)
                     for job in self.UC_job]
                )
            ]
        else:
            job = self.Tard_job[
                np.argmax(
                    [self.T_cur + job.estimated_t_ij - job.D
                     for job in self.Tard_job]
                )
            ]
        # Select operation
        operation = job.operations[job.OP]
        # Select machine
        machine = self.machines[
            np.argmin(
                [max(self.machines[idx].CTK, t_ij, job.A) if t_ij > -1 else MAXINT
                 for idx, t_ij in enumerate(operation.t_ijk)]
            )
        ]
        if operation.t_ijk[machine.idx] == -1:
            print("error rule1")
        return job.idx, machine.idx

    # Composite dispatching rule 2
    def rule2(self):
        # Select job
        if not self.Tard_job:
            job = self.UC_job[
                np.argmin(
                    [(job.D - self.T_cur) / job.estimated_t_ij
                     for job in self.UC_job]
                )
            ]
        else:
            job = self.Tard_job[
                np.argmax(
                    [self.T_cur + job.estimated_t_ij - job.D
                     for job in self.Tard_job]
                )
            ]
        # Select operation
        operation = job.operations[job.OP]
        # Select machine
        machine = self.machines[
            np.argmin(
                [max(self.machines[idx].CTK, t_ij, job.A) if t_ij > 0 else MAXINT
                 for idx, t_ij in enumerate(operation.t_ijk)]
            )
        ]
        if operation.t_ijk[machine.idx] == -1:
            print("error rule2")
        return job.idx, machine.idx

    # Composite dispatching rule 3
    def rule3(self):
        # Select job
        job = self.UC_job[
            np.argmax(
                [self.T_cur + job.estimated_t_ij - job.D
                 for job in self.UC_job]
            )
        ]
        # Select operation
        operation = job.operations[job.OP]
        # Select machine.
        if random.random() < 0.5:
            machine = self.machines[
                np.argmin(
                    [self.machines[idx].UK if t_ij > -1 else MAXINT
                     for idx, t_ij in enumerate(operation.t_ijk)]
                )
            ]
        else:
            machine = self.machines[
                np.argmin(
                    [self.machines[idx].CTK - self.machines[idx].ITK if t_ij > -1 else MAXINT
                     for idx, t_ij in enumerate(operation.t_ijk)]
                )
            ]
        if operation.t_ijk[machine.idx] == -1:
            print("error rule3")
        return job.idx, machine.idx

    # Composite dispatching rule 4
    def rule4(self):
        # Select job.
        job = self.UC_job[random.randint(0, len(self.UC_job)-1)]
        # Select operation
        operation = job.operations[job.OP]
        # Select machine
        machine = self.machines[
            np.argmin(
                [max(self.machines[idx].CTK, t_ij, job.A) if t_ij > -1 else MAXINT
                 for idx, t_ij in enumerate(operation.t_ijk)]
            )
        ]
        if operation.t_ijk[machine.idx] == -1:
            print("error rule4")
        return job.idx, machine.idx

    # Composite dispatching rule 5
    def rule5(self):
        # Select job.
        if not self.Tard_job:
            job = self.UC_job[
                np.argmin(
                    [job.CRJ * (job.D - self.T_cur)
                     for job in self.UC_job]
                )
            ]
        else:
            job = self.Tard_job[
                np.argmax(
                    [job.CRJ * (self.T_cur + job.estimated_t_ij - job.D)
                     for job in self.Tard_job]
                )
            ]
        # Select operation
        operation = job.operations[job.OP]
        # Select machine
        machine = self.machines[
            np.argmin(
                [max(self.machines[idx].CTK, t_ij, job.A) if t_ij > -1 else MAXINT
                 for idx, t_ij in enumerate(operation.t_ijk)]
            )
        ]
        if operation.t_ijk[machine.idx] == -1:
            print("error rule5")
        return job.idx, machine.idx

    # Composite dispatching rule 6
    def rule6(self):
        # Select job
        job = self.UC_job[
            np.argmax(
                [self.T_cur + job.estimated_t_ij - job.D
                 for job in self.UC_job]
            )
        ]
        # Select operation
        operation = job.operations[job.OP]
        # Select machine
        machine = self.machines[
            np.argmin(
                [max(self.machines[idx].CTK, t_ij, job.A) if t_ij > -1 else MAXINT
                 for idx, t_ij in enumerate(operation.t_ijk)]
            )
        ]
        if operation.t_ijk[machine.idx] == -1:
            print("error rule6")
        return job.idx, machine.idx

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
        print(f"Arrival time: {[job.A for job in self.jobs]}")
        print(f"Due date time: {[job.D for job in self.jobs]}")


if __name__ == "__main__":
    jsp = JobShopProblem(Param())
    for i in range(700):
        jsp.scheduling(jsp.rule2(), 2)
    print("Created!")

