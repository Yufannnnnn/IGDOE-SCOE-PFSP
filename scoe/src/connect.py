"""
滞留计数切换控制器与IG算法的连接接口

实现基于滞留计数的固定顺序切换机制与Iterated Greedy算法的集成，
当连续100次迭代未改进best_makespan时，切换到下一个算子（循环）。
"""

import time
import random
import logging
import math
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .ig import IG


# 默认配置（可在文件顶部方便地修改）
DEFAULT_CONFIG = {
    # stagnation_threshold 已改为动态计算，使用公式 τ = floor(1000/sqrt(n_jobs))
    'temperature_factor': 0.4
}


class FRRMAB_IG_Connector:
    """
    滞留计数切换控制器与IG算法的连接器
    
    负责管理基于滞留计数的固定顺序切换机制与IG算法的交互。
    算子按加入顺序固定序号，当连续100次迭代未改进best_makespan时，
    自动切换到下一个算子（循环回到第一个）。
    """
    
    def __init__(self, 
                 processing_times: np.ndarray,
                 n_jobs: int, 
                 n_machines: int,
                 # 配置参数
                 stagnation_threshold: Optional[int] = None,
                 temperature_factor: Optional[float] = None,
                 random_seed: Optional[int] = None,
                 # 仅支持：直接传入的算子列表（包含初始算子和候选算子）
                 operator_list: Optional[List] = None):
        """
        初始化滞留计数切换控制器-IG连接器
        
        Args:
            processing_times: 处理时间矩阵
            n_jobs: 工件数量
            n_machines: 机器数量
            stagnation_threshold: 滞留阈值（默认使用公式计算：floor(1000/sqrt(n_jobs))）
            temperature_factor: 模拟退火温度因子T（默认使用DEFAULT_CONFIG中的值）
            random_seed: 随机种子
            operator_list: 直接传入的算子列表（已适配破坏接口的对象），包含初始算子和候选算子
        """
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 配置参数处理：使用传入参数或默认值
        # stagnation_threshold: 如果未指定，使用公式 τ = floor(1000/sqrt(n_jobs)) 计算
        if stagnation_threshold is None:
            self.stagnation_threshold = int(math.floor(1000 / math.sqrt(n_jobs)))
        else:
            self.stagnation_threshold = stagnation_threshold
        self.temperature_factor = temperature_factor if temperature_factor is not None else DEFAULT_CONFIG['temperature_factor']
        
        # 参数验证
        if self.stagnation_threshold <= 0:
            raise ValueError(f"stagnation_threshold必须大于0，当前值: {self.stagnation_threshold}")
        if self.temperature_factor <= 0:
            raise ValueError(f"temperature_factor必须大于0，当前值: {self.temperature_factor}")
        
        # 保存processing_times用于温度计算
        self.processing_times = processing_times
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        # 预计算Temperature（与实例规模与处理时间相关）
        # Temperature = T * (sum_ij p_ij) / (n_jobs * n_machines * 10)
        total_processing_time = float(self.processing_times.sum())
        self.temperature = self.temperature_factor * (
            total_processing_time / (self.n_jobs * self.n_machines * 10.0)
        )
        
        # 初始化IG算法（不需要d_max参数，让破坏算子自主决定）
        self.ig = IG(
            processing_times=processing_times,
            n_jobs=n_jobs,
            n_machines=n_machines
        )
        
        # 根据传入的算子列表构建算子池（仅支持该方式）
        if operator_list is None or len(operator_list) == 0:
            raise ValueError("算子池不能为空，请通过 operator_list 传入已适配算子")
        self.destruction_operators = operator_list
        
        if len(self.destruction_operators) == 0:
            raise ValueError("算子池不能为空，请提供至少一个算子")
        
        self.num_operators = len(self.destruction_operators)
        
        # 滞留计数控制器状态
        self.current_operator_id = 0  # 当前使用的算子序号（从0开始）
        self.stagnation_count = 0  # 连续未改进迭代次数
        
        # 算法状态
        self.current_sequence = None
        self.best_sequence = None
        self.best_makespan = float('inf')
        self.iteration_count = 0
        
        # 统计信息
        self.operator_usage_stats = [0] * self.num_operators
        self.operator_success_stats = [0] * self.num_operators
        # 记录每次迭代使用的算子ID（用于详细日志）
        self.iteration_operator_ids = []
        # 记录每次迭代的best_makespan历史（用于绘制收敛曲线）
        self.best_makespan_history = []
        
        # 算子切换统计
        self.operator_switch_count = 0  # 总共切换算子的次数
        
        # 日志配置 - 设置为WARNING级别，减少详细输出
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def _execute_ig_iteration(self, operator_id: int, start_time=None, actual_time_limit=None) -> Tuple[List[int], float]:
        """
        执行一次完整的IG迭代
        
        Args:
            operator_id: 选中的破坏算子ID
            start_time: 开始时间
            actual_time_limit: 实际时间限制
            
        Returns:
            (新生成的解序列, makespan)
        """
        operator = self.destruction_operators[operator_id]
        
        # 步骤B: 破坏阶段
        remaining_sequence, removed_jobs = operator.destroy(
            self.current_sequence, 
            self.ig.processing_times
        )
        
        # 步骤C: 构造阶段（传入时间参数和fallback_sequence，与ig_pap2.py保持一致）
        new_sequence = self.ig.construction_phase(
            remaining_sequence, 
            removed_jobs,
            start_time=start_time,
            actual_time_limit=actual_time_limit,
            fallback_sequence=self.current_sequence  # 传入fallback，超时时返回完整解
        )
        
        # 步骤D: 局部搜索阶段（传入时间参数，与ig_pap2.py保持一致）
        new_sequence, new_makespan = self.ig.local_search_insertion(
            new_sequence,
            start_time=start_time,
            actual_time_limit=actual_time_limit
        )
        
        return new_sequence, new_makespan
    
    def solve(self, 
              time_limit: Optional[float] = None,
              verbose: bool = True) -> Tuple[List[int], float, Dict[str, Any]]:
        """
        使用滞留计数切换控制器-IG算法求解PFSP问题
        
        算子按固定序号顺序使用，当连续100次迭代未改进best_makespan时，
        自动切换到下一个算子（循环回到第一个）。
        
        Args:
            time_limit: 时间限制（秒）
            verbose: 是否输出详细信息
            
        Returns:
            best_sequence: 最优解序列
            best_makespan: 最优makespan
            stats: 算法统计信息
        """
        start_time = time.time()
        # 计算actual_time_limit（与ig_pap2.py保持一致，预留1%缓冲）
        actual_time_limit = time_limit * 0.99 if time_limit is not None else None
        
        # 初始化：使用NEH启发式生成初始解（与ig_pap2.py保持一致）
        self.current_sequence, current_makespan = self.ig.neh_heuristic()
        # 对初始解进行局部搜索
        self.current_sequence, current_makespan = self.ig.local_search_insertion(
            self.current_sequence,
            start_time=start_time,
            actual_time_limit=actual_time_limit
        )
        self.best_sequence = self.current_sequence.copy()
        self.best_makespan = current_makespan
        # 记录初始解
        self.best_makespan_history.append(self.best_makespan)
        
        # 主迭代循环（从算子0开始，不再需要冷启动）
        iteration = 0
        
        while True:
            # 检查停止条件（使用actual_time_limit，与ig_pap2.py保持一致）
            if actual_time_limit is not None:
                current_time = time.time()
                if (current_time - start_time) >= actual_time_limit:
                    break
            
            # 使用当前固定序号的算子
            selected_operator_id = self.current_operator_id
            # 记录本次迭代使用的算子ID
            self.iteration_operator_ids.append(selected_operator_id)
            
            # 执行IG迭代（传入时间参数，与ig_pap2.py保持一致）
            new_sequence, new_makespan = self._execute_ig_iteration(
                selected_operator_id,
                start_time=start_time,
                actual_time_limit=actual_time_limit
            )
                  
            # 模拟退火接受准则（与ig_pap2.py保持一致）
            # 计算Δ = C_max(π'') - C_max(π)，其中π''是新解，π是旧解
            delta = new_makespan - current_makespan
            
            # 先检查是否改进了best_makespan（在更新之前，用于滞留计数和统计）
            improved = new_makespan < self.best_makespan
            
            # 模拟退火接受准则
            if delta <= 0:
                # 如果Δ ≤ 0（新解更好），总是接受
                self.current_sequence = new_sequence.copy()
                # 在接受新解后，检查并更新best_makespan（与ig_pap2.py保持一致）
                if improved:
                    self.best_sequence = new_sequence.copy()
                    self.best_makespan = new_makespan
            else:
                # 否则，以概率exp(-Δ / Temperature)接受（使用预计算的温度）
                acceptance_probability = math.exp(-delta / self.temperature)
                if random.random() < acceptance_probability:
                    self.current_sequence = new_sequence.copy()
            
            # 根据是否改进了best_makespan进行滞留计数和算子切换
            if improved:
                # 有改进：重置滞留计数
                self.stagnation_count = 0
            else:
                # 未改进：增加滞留计数
                self.stagnation_count += 1
                
                # 检查是否需要切换算子
                if self.stagnation_count >= self.stagnation_threshold:
                    # 切换到下一个算子（循环）
                    self.current_operator_id = (self.current_operator_id + 1) % self.num_operators
                    self.stagnation_count = 0  # 重置滞留计数
                    self.operator_switch_count += 1  # 增加切换计数
            
            # 记录每次迭代的best_makespan（无论是否改进）
            self.best_makespan_history.append(self.best_makespan)
            
            # 更新统计信息
            self.operator_usage_stats[selected_operator_id] += 1
            if improved:
                self.operator_success_stats[selected_operator_id] += 1
            
            iteration += 1
            self.iteration_count = iteration
            # 移除详细的迭代日志输出
        
        # 计算最终统计信息
        total_time = time.time() - start_time
        stats = self._compute_statistics(total_time)
        
        # 移除详细的完成日志输出
        
        return self.best_sequence, self.best_makespan, stats
    
    def _compute_statistics(self, total_time: float) -> Dict[str, Any]:
        """计算算法统计信息"""
        # 计算算子成功率
        success_rates = []
        for i in range(self.num_operators):
            if self.operator_usage_stats[i] > 0:
                success_rate = self.operator_success_stats[i] / self.operator_usage_stats[i]
            else:
                success_rate = 0.0
            success_rates.append(success_rate)
        
        return {
            'total_iterations': self.iteration_count,
            'total_time': total_time,
            'best_makespan': self.best_makespan,
            'operator_usage': self.operator_usage_stats.copy(),
            'operator_success': self.operator_success_stats.copy(),
            'operator_success_rates': success_rates,
            'current_operator_id': self.current_operator_id,
            'stagnation_count': self.stagnation_count,
            'operator_switch_count': self.operator_switch_count,  # 算子切换总次数
            'iteration_operator_ids': self.iteration_operator_ids.copy(),  # 每次迭代使用的算子ID列表
            'best_makespan_history': self.best_makespan_history.copy(),  # 每次改进的best_makespan历史
            # 配置参数（方便查看使用的参数）
            'config': {
                'stagnation_threshold': self.stagnation_threshold,
                'temperature_factor': self.temperature_factor
            }
        }
    
    def get_operator_performance(self) -> Dict[str, Any]:
        """获取算子性能分析"""
        performance = {}
        
        for i, operator in enumerate(self.destruction_operators):
            usage = self.operator_usage_stats[i]
            success = self.operator_success_stats[i]
            success_rate = success / usage if usage > 0 else 0.0
            
            performance[f'operator_{i}'] = {
                'name': operator.name,
                'usage_count': usage,
                'success_count': success,
                'success_rate': success_rate
            }
        
        return performance
    
    def reset(self) -> None:
        """重置算法状态"""
        self.current_operator_id = 0
        self.stagnation_count = 0
        self.current_sequence = None
        self.best_sequence = None
        self.best_makespan = float('inf')
        self.iteration_count = 0
        self.operator_usage_stats = [0] * self.num_operators
        self.operator_success_stats = [0] * self.num_operators
        self.iteration_operator_ids = []
        self.best_makespan_history = []
        self.operator_switch_count = 0  # 重置切换计数




def create_frrmab_ig_solver(processing_times: np.ndarray,
                           n_jobs: int,
                           n_machines: int,
                           **kwargs) -> FRRMAB_IG_Connector:
    """
    创建滞留计数切换控制器-IG求解器的便捷函数
    """
    return FRRMAB_IG_Connector(
        processing_times=processing_times,
        n_jobs=n_jobs,
        n_machines=n_machines,
        **kwargs
    )
