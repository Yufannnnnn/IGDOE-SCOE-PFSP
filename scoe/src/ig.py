import numpy as np
from typing import List, Tuple
import random
import time


def read_instance(file_path):
    """读取实例文件"""
    with open(file_path, 'r') as f:
        # 读取第一行获取作业数和机器数
        n_jobs, n_machines = map(int, f.readline().strip().split())
        
        # 初始化处理时间矩阵
        processing_times = np.zeros((n_jobs, n_machines), dtype=int)
        
        # 读取每个工件的处理时间
        for i in range(n_jobs):
            # 读取一行数据，每行包含机器-时间对
            pairs = list(map(int, f.readline().strip().split()))
            # 将机器-时间对转换为处理时间数组
            for j in range(0, len(pairs), 2):
                machine = pairs[j]
                time = pairs[j + 1]
                processing_times[i][machine] = time
                
    return processing_times, n_jobs, n_machines


class IG(object):
    def __init__(self, processing_times: np.ndarray, n_jobs: int, n_machines: int,
                 d_max: int = 4, max_iterations: int = 50, destroy_module_name: str = None):
        self.processing_times = processing_times
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.d_max = d_max
        self.max_iterations = max_iterations
        self.destroy_module_name = destroy_module_name
        



    def calculate_objectives(self, sequence: List[int]) -> Tuple[float, List[List[Tuple[float, float]]]]:
        """计算makespan和调度信息，只在算法的最终结果返回时使用"""
        # 直接调用calculate_makespan计算makespan
        makespan = self.calculate_makespan(sequence)
        
        # 以下仅计算详细调度信息
        schedules = []
        machine_completion = [0] * self.n_machines

        for job in sequence:
            job_schedule = []
            for m in range(self.n_machines):
                start_time = machine_completion[m]
                if m > 0:
                    start_time = max(start_time, job_schedule[m - 1][1])
                end_time = start_time + self.processing_times[job][m]
                job_schedule.append((start_time, end_time))
                machine_completion[m] = end_time
            schedules.append(job_schedule)

        return makespan, schedules

    def calculate_makespan(self, sequence: List[int]) -> float:
        """计算makespan"""
        n_jobs = len(sequence)
        n_machines = self.n_machines
        
        # 初始化评估矩阵
        e = np.zeros((n_jobs+1, n_machines+1))
        
        # 计算e值（最早完成时间）
        for i in range(1, n_jobs+1):
            for j in range(1, n_machines+1):
                e[i, j] = max(e[i, j-1], e[i-1, j]) + self.processing_times[sequence[i-1]][j-1]
        
        makespan = e[n_jobs, n_machines]
        return makespan

    def _compute_forward_times(self, sequence):
        """计算前向完成时间 E[i][j] = 前i个工件在机器j上的完成时间"""
        k = len(sequence)
        m = self.n_machines
        E = np.zeros((k + 1, m + 1))  # E[0][j] = 0, E[i][0] = 0

        for i in range(1, k + 1):
            job = sequence[i - 1]  # 第i个位置的工件（0-based索引）
            for j in range(1, m + 1):
                # E[i][j] = max(前一个工件在当前机器的完成时间, 当前工件在前一台机器的完成时间) + 当前处理时间
                E[i][j] = max(E[i - 1][j], E[i][j - 1]) + self.processing_times[job][j - 1]
        return E

    def _compute_backward_times(self, sequence):
        """计算反向尾部时间 Q[i][j] = 从位置i开始到序列结束在机器j上需要的时间"""
        k = len(sequence)
        m = self.n_machines
        # Q[k+1][j] = 0 (序列结束), Q[i][m+1] = 0 (最后一台机器之后)
        Q = np.zeros((k + 2, m + 2))

        # 从后往前计算
        for i in range(k, 0, -1):
            job = sequence[i - 1]  # 位置i对应的工件
            for j in range(m, 0, -1):
                # Q[i][j] = max(下一个位置在当前机器的时间, 当前位置在下一台机器的时间) + 当前处理时间
                Q[i][j] = max(Q[i + 1][j], Q[i][j + 1]) + self.processing_times[job][j - 1]
        return Q

    def _insert_and_evaluate(self, E, Q, sequence, job_to_insert, insert_pos):
        """评估在指定位置插入工件后的makespan"""
        m = self.n_machines
        # F[j] = 插入工件在机器j上的完成时间
        F = np.zeros(m + 1)
        
        # 计算插入工件在每台机器上的完成时间
        for j in range(1, m + 1):
            # max(前面工件在当前机器的完成时间, 当前工件在前一台机器的完成时间) + 处理时间
            F[j] = max(E[insert_pos][j], F[j - 1]) + self.processing_times[job_to_insert][j - 1]

        # 计算makespan = max(插入工件的完成时间 + 后续工件的尾部时间)
        makespan = 0
        for j in range(1, m + 1):
            # 添加边界检查
            if insert_pos + 1 < Q.shape[0]:
                makespan = max(makespan, F[j] + Q[insert_pos + 1][j])
            else:
                makespan = max(makespan, F[j])  # 如果插在最后，没有尾部时间
        return makespan

    def neh_heuristic(self) -> Tuple[List[int], float]:
        """使用NEH启发式算法"""
        # 按总处理时间降序排列作业
        job_order = np.argsort(-self.processing_times.sum(axis=1)).tolist()
        sequence = [job_order[0]]
        
        # 初始序列只有一个工件，计算其makespan
        final_makespan = self.calculate_makespan(sequence)

        for k in range(1, self.n_jobs):
            job = job_order[k]
            best_makespan = float('inf')
            best_position = 0

            # 计算当前序列的前向完成时间和反向尾部时间
            E = self._compute_forward_times(sequence)
            Q = self._compute_backward_times(sequence)

            # 尝试所有可能的插入位置
            for i in range(len(sequence) + 1):
                ms = self._insert_and_evaluate(E, Q, sequence, job, i)
                if ms < best_makespan:
                    best_makespan = ms
                    best_position = i

            sequence.insert(best_position, job)
            final_makespan = best_makespan  # 更新最终makespan

        return sequence, final_makespan

    def construction_phase(self, sequence: List[int], candidates: List[int], start_time=None, actual_time_limit=None, fallback_sequence=None) -> List[int]:
        """
        构造阶段：使用插入法重新构建序列，增加时间检查。
        超时时直接返回 fallback_sequence（即破坏前的完整解）。
        """
        seq = sequence.copy()
        for job in candidates:
            # 超时检查
            if actual_time_limit is not None and start_time is not None:
                if (time.time() - start_time) >= actual_time_limit:
                    return fallback_sequence if fallback_sequence is not None else seq
            best_makespan = float('inf')
            best_position = 0
            E = self._compute_forward_times(seq)
            Q = self._compute_backward_times(seq)
            for i in range(len(seq) + 1):
                ms = self._insert_and_evaluate(E, Q, seq, job, i)
                if ms < best_makespan:
                    best_makespan = ms
                    best_position = i
            seq.insert(best_position, job)
        return seq

    def local_search_insertion(self, sequence: List[int], start_time=None, actual_time_limit=None) -> Tuple[List[int], float]:
        seq = sequence.copy()
        E = self._compute_forward_times(seq)
        current_makespan = E[len(seq)][self.n_machines]
        for i in range(len(seq)):
            # 时间检查
            if actual_time_limit is not None and start_time is not None:
                if (time.time() - start_time) >= actual_time_limit:
                    return seq, current_makespan  # 这里 seq 一定是完整的
            job = seq[i]
            seq.pop(i)
            E_removed = self._compute_forward_times(seq)
            Q_removed = self._compute_backward_times(seq)
            best_pos = i
            best_makespan = current_makespan
            for j in range(len(seq) + 1):
                makespan = self._insert_and_evaluate(E_removed, Q_removed, seq, job, j)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_pos = j
            seq.insert(best_pos, job)
            if best_makespan < current_makespan:
                return seq, best_makespan
            current_makespan = best_makespan
        return seq, current_makespan

