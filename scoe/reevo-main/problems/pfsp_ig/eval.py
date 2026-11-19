import sys
import os
import numpy as np
from os import path
import logging
import time
import importlib
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

# 添加项目根目录和src目录到sys.path
current_dir = path.dirname(path.abspath(__file__))
project_root = path.normpath(path.join(current_dir, '../../..'))
src_dir = path.join(project_root, 'src')
reevo_output_dir = path.join(project_root, 'reevo-main', 'output_algorithms')

# 添加路径到sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if reevo_output_dir not in sys.path:
    sys.path.insert(0, reevo_output_dir)

# 导入必要的模块 - 使用绝对导入
try:
    from src.ig import IG, read_instance
    from src.connect import create_frrmab_ig_solver
    print("所有模块导入成功")
except ImportError as e:
    print(f"导入模块失败: {e}")

# 全局缓存算子池，避免重复构建
_cached_operator_pool = None
_cached_operator_pool_key = None

# 全局缓存UB字典，避免重复读取Excel
_cached_ub_dict = None

def get_operator_pool():
    """获取算子池，使用缓存避免重复构建"""
    global _cached_operator_pool, _cached_operator_pool_key
    
    # 检查算子池环境变量是否发生变化
    current_key = os.getenv('AUTOREEVO_OPERATOR_POOL', 'initial_operator')
    
    if _cached_operator_pool is None or _cached_operator_pool_key != current_key:
        _cached_operator_pool = build_operator_pool()
        _cached_operator_pool_key = current_key
    
    return _cached_operator_pool

def solve_instance_with_frrmab(file_path: str, time_limit: float = None) -> float:
    """
    使用滞留计数切换控制器+IG算法求解PFSP实例（与算子池组合评估）
    
    参数:
        file_path: 实例文件路径
        time_limit: 限制运行时间（秒），如果为None则自动根据实例规模计算
        
    返回:
        makespan: 最大完工时间
    """
    import time
    
    # 读取实例文件第一行，获取n_jobs和n_machines
    with open(file_path, 'r') as f:
        first_line = f.readline()
        n_jobs, n_machines = map(int, first_line.strip().split())
    if time_limit is None:
        time_limit = n_jobs * n_machines * 0.06
    
    processing_times, n_jobs, n_machines = read_instance(file_path)
    
    # 设置日志级别
    logging.basicConfig(level=logging.WARNING)
    
    # 使用缓存的算子池
    operator_list = get_operator_pool()
    
    # 使用滞留计数切换控制器+IG求解（失败时让异常外抛，不做回退）
    solver = create_frrmab_ig_solver(
        processing_times=processing_times,
        n_jobs=n_jobs,
        n_machines=n_machines,
        operator_list=operator_list
    )
    
    _, makespan, _ = solver.solve(time_limit=time_limit, verbose=False)
    return float(makespan)

def build_operator_pool():
    """
    构建完整的算子池：已有算子 + 当前候选算子
    """
    operator_list = []
    
    # 1. 从算子池目录加载已有算子
    operator_pool_str = os.getenv('AUTOREEVO_OPERATOR_POOL', '')
    operator_pool_dir = os.getenv('AUTOREEVO_OPERATOR_POOL_DIR', '')
    
    if operator_pool_str and operator_pool_dir and os.path.exists(operator_pool_dir):
        # 解析算子池模块名
        operator_names = [name.strip() for name in operator_pool_str.split(',')]
        
        for name in operator_names:
            try:
                # 从算子池目录加载算子
                module_path = os.path.join(operator_pool_dir, f"{name}.py")
                if os.path.exists(module_path):
                    # 在函数内部重新导入importlib.util
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 创建算子适配器
                    operator = OperatorAdapter(module, name)
                    operator_list.append(operator)
                    print(f"已加载算子: {name}")
                else:
                    print(f"算子文件不存在: {module_path}")
            except Exception as e:
                print(f"加载算子 {name} 失败: {e}")
    
    # 2. 添加当前候选算子（从gpt.py文件）
    try:
        current_candidate = load_current_candidate_operator()
        if current_candidate:
            operator_list.append(current_candidate)
    except Exception as e:
        print(f"加载当前候选算子失败: {e}")
    
    # 只在第一次构建时输出日志
    if not hasattr(build_operator_pool, '_logged'):
        print(f"构建算子池完成，包含 {len(operator_list)} 个算子")
        build_operator_pool._logged = True
    
    return operator_list

def load_current_candidate_operator():
    """
    加载当前候选算子（从gpt.py文件）
    """
    try:
        # 读取gpt.py文件
        gpt_file = os.path.join(project_root, 'reevo-main', 'problems', 'pfsp_ig', 'gpt.py')
        if not os.path.exists(gpt_file):
            return None
        
        # 创建临时模块
        with open(gpt_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        # 动态导入 - 在函数内部重新导入importlib.util
        import importlib.util
        spec = importlib.util.spec_from_file_location("current_candidate", temp_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 清理临时文件
        os.unlink(temp_file_path)
        
        # 创建算子适配器
        return OperatorAdapter(module, "current_candidate")
        
    except Exception as e:
        # 只在第一次失败时输出错误信息
        if not hasattr(load_current_candidate_operator, '_error_logged'):
            print(f"加载当前候选算子失败: {e}")
            load_current_candidate_operator._error_logged = True
        return None

class OperatorAdapter:
    """
    算子适配器，将外部模块适配为破坏算子接口
    """
    def __init__(self, module, name_hint: str):
        self._module = module
        self.name = getattr(module, 'name', name_hint)
    
    def destroy(self, sequence, processing_times):
        # ReEvo生成的所有算子都使用 destroy_v2 接口；缺失或执行失败应直接抛异常
        if hasattr(self._module, 'destroy_v2') and callable(self._module.destroy_v2):
            return self._module.destroy_v2(sequence, processing_times)
        else:
            raise RuntimeError(f"算子 {self.name} 缺少 destroy_v2 方法")

def load_ub_dict():
    """
    从Excel文件读取UB值，创建实例名到UB的映射字典（同时支持 Small 与 Large）
    字典键格式：'Small/VFR40_5_1_Gap.txt' 或 'Large/VFR100_20_1_Gap.txt'
    使用全局缓存避免重复读取
    """
    global _cached_ub_dict
    
    if _cached_ub_dict is not None:
        return _cached_ub_dict
    
    current_dir = path.dirname(path.abspath(__file__))
    excel_path = path.join(current_dir, 'VRF_Instances', 'BestSolutionsAndBounds.xlsx')
    
    if not path.exists(excel_path):
        raise FileNotFoundError(f"UB文件不存在: {excel_path}")
    
    try:
        # 同时读取两个 sheet
        df_large = pd.read_excel(excel_path, sheet_name='Large')
        df_small = pd.read_excel(excel_path, sheet_name='Small')
        
        # 创建映射字典：带目录前缀的文件相对路径 -> UB值
        ub_dict = {}
        # Large: Excel中的实例名没有_Gap，需要添加_Gap.txt
        for _, row in df_large.iterrows():
            instance_name = row['Instance']  # 例如: VFR100_20_1（没有_Gap）
            ub = row['UB']
            file_key = f"Large/{instance_name}_Gap.txt"  # 添加_Gap.txt后缀
            ub_dict[file_key] = ub
        # Small: Excel中的实例名已有_Gap，只需添加.txt
        for _, row in df_small.iterrows():
            instance_name = row['Instance']  # 例如: VFR40_5_1_Gap（已有_Gap）
            ub = row['UB']
            file_key = f"Small/{instance_name}.txt"  # 只需添加.txt后缀
            ub_dict[file_key] = ub
        
        _cached_ub_dict = ub_dict
        print(f"成功加载UB值：Large {len(df_large)} 条，Small {len(df_small)} 条，总计 {len(ub_dict)} 条")
        return ub_dict
        
    except Exception as e:
        raise RuntimeError(f"读取UB文件失败: {e}")

def solve_instance(file_path: str, time_limit: float = None) -> float:
    """
    使用IG算法求解PFSP实例（原始方法，作为回退）
    
    参数:
        file_path: 实例文件路径
        time_limit: 限制运行时间（秒），如果为None则自动根据实例规模计算
        
    返回:
        makespan: 最大完工时间
    """
    import time
    # 读取实例文件第一行，获取n_jobs和n_machines
    with open(file_path, 'r') as f:
        first_line = f.readline()
        n_jobs, n_machines = map(int, first_line.strip().split())
    if time_limit is None:
        time_limit = n_jobs * n_machines * 0.06
    processing_times, n_jobs, n_machines = read_instance(file_path)
    
    # 不需要太详细的日志，使用最简单的日志设置
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger('IG')
    
    ig_solver = IG(processing_times, n_jobs, n_machines)
    
    # 求解实例 - 使用NEH启发式
    try:
        sequence, makespan = ig_solver.neh_heuristic()
        return float(makespan)
    except Exception as e:
        print(f"NEH启发式失败: {e}")
        return float('inf')

def evaluate_specific_instances(instance_files: list) -> float:
    """
    评估指定实例文件的性能，每个算例运行五次取平均值，然后计算相对误差
    支持传入 'Small/xxx_Gap.txt' 与 'Large/xxx_Gap.txt' 混合列表
    返回实例列表的平均相对误差
    """
    import time
    current_dir = path.dirname(path.abspath(__file__))
    # 根目录：VRF_Instances（子目录含 Small/ 与 Large/）
    instances_root = path.join(current_dir, 'VRF_Instances')
    
    # 加载UB字典
    ub_dict = load_ub_dict()
    
    tasks = []
    for instance in instance_files:
        # instance 示例：'Small/VFR40_5_1_Gap.txt' 或 'Large/VFR100_20_1_Gap.txt'
        instance_path = path.join(instances_root, instance)
        for run_id in range(1, 6):
            tasks.append((instance, instance_path, run_id))

    results = {}
    # 每个进程负责一个solve_instance，即某个实例的某一次运行
    with ProcessPoolExecutor(max_workers=10) as executor:
        future_to_task = {
            executor.submit(solve_instance_with_frrmab, instance_path): (instance, run_id)
            for (instance, instance_path, run_id) in tasks
        }
        for future in as_completed(future_to_task):
            instance, run_id = future_to_task[future]
            makespan = future.result()
            if instance not in results:
                results[instance] = []
            results[instance].append(makespan)
            print(f"实例 {instance} 运行 {run_id}: {makespan}")

    # 计算每个实例的平均makespan和相对误差
    relative_errors = []
    for instance in instance_files:
        if instance not in results:
            print(f"警告: 实例 {instance} 没有结果，跳过")
            continue
        
        runs = results[instance]
        avg_makespan = np.mean(runs)
        
        # 从UB字典中查找对应实例的UB值（带前缀键）
        if instance not in ub_dict:
            print(f"警告: 实例 {instance} 在UB字典中未找到，跳过")
            continue
        
        ub = ub_dict[instance]
        
        # 计算相对误差: (avg_makespan - UB) / UB
        relative_error = (avg_makespan - ub) / ub
        relative_errors.append(relative_error)
        
        print(f"实例 {instance}: 平均makespan={avg_makespan:.2f}, UB={ub}, 相对误差={relative_error:.6f} ({relative_error*100:.4f}%)")
    
    if not relative_errors:
        raise ValueError("没有有效的相对误差数据，所有实例都被跳过")
    
    # 返回平均相对误差
    avg_relative_error = np.mean(relative_errors)
    print(f"平均相对误差: {avg_relative_error:.6f} ({avg_relative_error*100:.4f}%)")
    return avg_relative_error


if __name__ == "__main__":
    print("[*] Running ...")
    
    # 使用VRF_Instances/Large目录下的9个VRF100算例
    '''
    # 原Ta实例（已注释）：
    selected_instances = [
        'ta061',
        'ta062',
        'ta071',
        'ta072',
        'ta081',
        'ta082',
        'ta091',
        'ta092',
        'ta101',
        'ta102'
    ]
    '''
    selected_instances = [
        'Large/VFR100_20_1_Gap.txt',
        'Large/VFR100_20_3_Gap.txt',
        'Large/VFR100_20_5_Gap.txt',
        'Large/VFR100_40_2_Gap.txt',
        'Large/VFR100_40_4_Gap.txt',
        'Large/VFR100_40_6_Gap.txt',
        'Large/VFR100_60_9_Gap.txt',
        'Large/VFR200_40_10_Gap.txt',
        'Large/VFR200_60_8_Gap.txt',
     ]

    instance_count = len(selected_instances)
    print(f"使用 {instance_count} 个算例: {selected_instances}")
    
    # 直接执行训练逻辑：使用指定的实例，每个算例运行5次
    avg_relative_error = evaluate_specific_instances(selected_instances)
    print(f"[*] 训练集（{instance_count}个算例）平均相对误差: {avg_relative_error:.6f} ({avg_relative_error*100:.4f}%)")
    # 输出平均相对误差（ReEvo使用倒数第二行作为评估指标）
    print(f"[*] Average:")
    print(avg_relative_error)