import os
import sys
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from statistics import mean
from datetime import datetime
import random
import math
from typing import List, Tuple
import pandas as pd
import glob

# 全局变量定义（模块级别）
_run_output_dir = None
_logger_configured = False
logger = logging.getLogger(__name__)

def setup_logging():
    """配置日志系统，只在主进程中调用一次"""
    global _run_output_dir, _logger_configured
    
    # 防止重复配置
    if _logger_configured:
        return _run_output_dir
    
    # 创建运行目录和日志文件
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _output_base_dir = os.path.normpath(os.path.join(_current_dir, './output'))
    _run_output_dir = os.path.join(_output_base_dir, f'run_{_timestamp}')
    os.makedirs(_run_output_dir, exist_ok=True)
    
    _logs_dir = os.path.join(_run_output_dir, 'logs')
    os.makedirs(_logs_dir, exist_ok=True)
    _log_file = os.path.join(_logs_dir, f"run_{_timestamp}.log")
    
    # 配置日志处理器
    _file_handler = logging.FileHandler(_log_file, encoding='utf-8')
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 清除所有现有处理器，避免重复
    _root_logger = logging.getLogger()
    _root_logger.handlers.clear()
    _root_logger.addHandler(_file_handler)
    _root_logger.setLevel(logging.INFO)
    
    _logger_configured = True
    logger.info(f"运行输出目录: {_run_output_dir}")
    
    return _run_output_dir

MAX_OP_SIZE = 5  # 最终算子池大小：1个初始算子 + 4轮训练 = 5个算子
NUM_CANDIDATES = 10  # 每轮生成的候选算法数量

class Algorithm:
    def __init__(self, name, code, path):
        self.name = name
        self.code = code
        self.path = path

# 缓存和无效候选跟踪
RESULTS_CACHE = {}  # { key(算子名称元组): { 实例路径: 相对误差 } }
INVALID_CANDIDATES = set()  # 无效候选算法集合
INVALID_CANDIDATES_COUNT = {}  # 每代无效候选算法计数

# 全局缓存UB字典，避免重复读取Excel
_cached_ub_dict = None

def load_ub_dict():
    """
    从Excel文件读取UB值，创建实例文件名到UB的映射字典（同时支持 Small 与 Large）
    字典键格式：'Small/VFR40_5_1_Gap.txt' 或 'Large/VFR100_20_1_Gap.txt'
    使用全局缓存避免重复读取
    """
    global _cached_ub_dict
    
    if _cached_ub_dict is not None:
        return _cached_ub_dict
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(current_dir, 'reevo-main', 'problems', 'pfsp_ig', 'VRF_Instances', 'BestSolutionsAndBounds.xlsx')
    
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"UB文件不存在: {excel_path}")
    
    try:
        # 同时读取两个 sheet
        df_large = pd.read_excel(excel_path, sheet_name='Large')
        df_small = pd.read_excel(excel_path, sheet_name='Small')
        
        ub_dict = {}
        
        # 填充 Large: Excel中的实例名没有_Gap，需要添加_Gap.txt
        for _, row in df_large.iterrows():
            instance_name = row['Instance']  # 例如: VFR100_20_1（没有_Gap）
            ub = row['UB']
            file_key = f"Large/{instance_name}_Gap.txt"  # 添加_Gap.txt后缀
            ub_dict[file_key] = ub
        
        # 填充 Small: Excel中的实例名已有_Gap，只需添加.txt
        for _, row in df_small.iterrows():
            instance_name = row['Instance']  # 例如: VFR40_5_1_Gap（已有_Gap）
            ub = row['UB']
            file_key = f"Small/{instance_name}.txt"  # 只需添加.txt后缀
            ub_dict[file_key] = ub
        
        _cached_ub_dict = ub_dict
        logger.info(f"成功加载UB值：Large {len(df_large)} 条，Small {len(df_small)} 条，总计 {len(ub_dict)} 条")
        return ub_dict
        
    except Exception as e:
        raise RuntimeError(f"读取UB文件失败: {e}")

def generate_algorithms(num_candidates, op_pool, generation=0, force_regenerate=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reevo_dir = os.path.normpath(os.path.join(current_dir, './reevo-main'))
    output_dir = os.path.normpath(os.path.join(reevo_dir, 'output_algorithms'))
    os.makedirs(output_dir, exist_ok=True)

    if force_regenerate:
        logger.info(f"调用ReEvo生成第{generation+1}代的{num_candidates}个候选算法...")
        try:
            original_cwd = os.getcwd()
            os.chdir(reevo_dir)
            cmd = [sys.executable, 'reevo_main.py']
            env = os.environ.copy()
            env['AUTOREEVO_PAP_GENERATION'] = str(generation)
            env['AUTOREEVO_PAP_CALL'] = '1'
            
            # 传递当前算子池信息给ReEvo
            if op_pool:
                # 使用完整的名称（包含_genX），不要split
                env['AUTOREEVO_OPERATOR_POOL'] = ','.join(op_pool)
                # 指向专门的算子池目录，并确保目录存在
                operator_pool_dir = os.path.join(_run_output_dir, 'operator_pool')
                os.makedirs(operator_pool_dir, exist_ok=True)
                env['AUTOREEVO_OPERATOR_POOL_DIR'] = operator_pool_dir
                logger.info(f"传递算子池信息给ReEvo: {env['AUTOREEVO_OPERATOR_POOL']}")
            else:
                # 如果没有算子池，只传递初始算子
                env['AUTOREEVO_OPERATOR_POOL'] = 'initial_operator'
                env['AUTOREEVO_OPERATOR_POOL_DIR'] = output_dir
                logger.info("传递初始算子信息给ReEvo")
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=100000)
            if result.returncode != 0:
                logger.error(f"ReEvo执行失败，返回码={result.returncode}")
                logger.error(f"错误输出: {result.stderr}")
                raise RuntimeError(result.stderr)
            logger.info("ReEvo执行完成")
            # 不再记录ReEvo的标准输出，保持日志简洁
        finally:
            os.chdir(original_cwd)
        time.sleep(2)
    else:
        logger.info(f"使用现有候选算法在目录: {output_dir}")

    candidates = []
    try:
        file_names = sorted([f for f in os.listdir(output_dir) if f.startswith('algorithm_') and f.endswith('.py')])
    except FileNotFoundError:
        logger.error(f"候选算法目录未找到: {output_dir}")
        file_names = []

    if not file_names:
        logger.warning(f"在目录{output_dir}中未找到候选算法文件")

    for fname in file_names[:num_candidates]:
        code_path = os.path.join(output_dir, fname)
        try:
            with open(code_path, "r", encoding="utf-8") as f:
                code = f.read()
            base_name = os.path.splitext(fname)[0]
            unique_name = f"{base_name}_gen{generation}"
            candidates.append(Algorithm(name=unique_name, code=code, path=code_path))
            logger.info(f"已加载 {fname} 作为 {unique_name}")
        except FileNotFoundError:
            logger.error(f"文件未找到: {code_path}")
            continue

    logger.info(f"已加载 {len(candidates)} 个候选算法")
    return candidates

def _solve_instance_with_frrmab(op_names, instance_path):
    """
    在给定算子池（破坏算子模块名称）上运行滞留计数切换控制器-IG算法求解一个实例。
    返回完工时间（浮点数）。
    """
    try:
        # 为src和output_algorithms设置sys.path
        repo_root = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.join(repo_root, 'src')
        out_ops_dir = os.path.normpath(os.path.join(repo_root, 'reevo-main', 'output_algorithms'))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        if out_ops_dir not in sys.path:
            sys.path.insert(0, out_ops_dir)

        from src.ig import read_instance
        from src.connect import create_frrmab_ig_solver
        import time as _time

        # 推导时间限制
        with open(instance_path, 'r') as f:
            first = f.readline()
            n_jobs, n_machines = map(int, first.strip().split())
        time_limit = n_jobs * n_machines * 0.06

        processing_times, n_jobs, n_machines = read_instance(instance_path)
        
        # 构建算子列表：从operator_pool_dir加载已有算子 + 从output_algorithms加载候选算子
        operator_list = []
        
        # 构建算子池目录路径
        current_run_dir = os.path.basename(_run_output_dir)
        operator_pool_dir = os.path.join(os.path.dirname(__file__), 'output', current_run_dir, 'operator_pool')
        
        # 统一加载所有算子（已有算子从operator_pool_dir加载，候选算子从output_algorithms加载）
        plugin_names = list(op_names)  # 包含所有算子名称，不再过滤initial_operator
        plugin_ops = _load_plugin_operators(plugin_names, operator_pool_dir, out_ops_dir)
        operator_list.extend(plugin_ops)
        
        # 创建求解器，使用算子列表
        solver = create_frrmab_ig_solver(
            processing_times=processing_times,
            n_jobs=n_jobs,
            n_machines=n_machines,
            operator_list=operator_list
        )
        _, makespan, _ = solver.solve(time_limit=float(time_limit), verbose=False)
        return float(makespan)
    except Exception as e:
        key = tuple(op_names)
        if key not in INVALID_CANDIDATES:
            INVALID_CANDIDATES.add(key)
            logger.warning(f"检测到无效的算子集合: {op_names} - {e}")
        return float('inf')


def _load_plugin_operators(module_names, operator_pool_dir, output_algorithms_dir):
    """从指定目录动态导入给定模块名，适配为破坏算子接口。
    
    加载逻辑：
    1. 优先从 operator_pool_dir 加载（已有算子）
    2. 如果找不到且是候选算子格式（algorithm_X_genY），则从 output_algorithms_dir 加载
    """
    import importlib.util
    import re
    loaded = []
    
    class _PluginOperatorAdapter:
        """将外部模块适配为破坏算子接口。"""
        def __init__(self, module, name_hint: str):
            self._module = module
            self.name = getattr(module, 'name', name_hint)
        
        def destroy(self, sequence, processing_times):
            # 优先使用 destroy_v2 接口，如果没有则尝试 destroy_v3（容错机制）
            if hasattr(self._module, 'destroy_v2') and callable(self._module.destroy_v2):
                try:
                    return self._module.destroy_v2(sequence, processing_times)
                except Exception as e:
                    logger.warning(f"算子 {self.name} 的 destroy_v2 执行失败: {e}")
                    return sequence, []
            elif hasattr(self._module, 'destroy_v3') and callable(self._module.destroy_v3):
                try:
                    logger.info(f"算子 {self.name} 没有 destroy_v2，使用 destroy_v3 作为容错")
                    return self._module.destroy_v3(sequence, processing_times)
                except Exception as e:
                    logger.warning(f"算子 {self.name} 的 destroy_v3 执行失败: {e}")
                    return sequence, []
            else:
                logger.warning(f"算子 {self.name} 既没有 destroy_v2 也没有 destroy_v3 方法")
                return sequence, []
    
    for mod_name in module_names:
        try:
            # 处理 initial_operator 的特殊映射（op_names中是"initial_operator"，但文件是"initial_operator_gen0.py"）
            if mod_name == "initial_operator":
                mod_name = "initial_operator_gen0"
            
            # 1. 优先从算子池目录加载（已有算子）
            module_path = os.path.join(operator_pool_dir, f"{mod_name}.py")
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(mod_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                loaded.append(_PluginOperatorAdapter(module, mod_name))
                continue
            
            # 2. 如果找不到，且名称匹配候选算子格式（algorithm_X_genY），则从 output_algorithms_dir 加载
            match = re.match(r'(algorithm_\d+)_gen\d+', mod_name)
            if match:
                base_name = match.group(1)  # 提取基础名称，如 algorithm_0_gen0 -> algorithm_0
                candidate_module_path = os.path.join(output_algorithms_dir, f"{base_name}.py")
                if os.path.exists(candidate_module_path):
                    spec = importlib.util.spec_from_file_location(mod_name, candidate_module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    loaded.append(_PluginOperatorAdapter(module, mod_name))
                else:
                    logger.warning(f"候选算子文件不存在: {candidate_module_path}")
            else:
                logger.warning(f"算子文件不存在: {module_path}")
        except Exception as e:
            logger.warning(f"加载插件算子失败: {mod_name} - {e}")
    return loaded

def evaluate_opset_on_instances(op_names, train_instances, repeats=20, max_workers=300, generation=0):
    """
    在所有实例上评估给定的算子集合（滞留计数切换控制器组合），重复指定次数。
    计算每个实例的平均makespan，然后计算相对误差。
    返回: { 实例路径: 相对误差 }
    """
    if not train_instances:
        return {}

    # 加载UB字典
    ub_dict = load_ub_dict()

    tasks = []
    for inst in train_instances:
        for _ in range(repeats):
            tasks.append((tuple(op_names), inst))

    runs = {}  # (算子元组, 实例) -> list[完工时间]
    logger.info(f"提交 {len(tasks)} 个滞留计数切换控制器运行任务: |OP|={len(op_names)} × {len(train_instances)} × 重复次数={repeats}")
    logger.info(f"算子池包含: {', '.join(op_names)}")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {}
        for key, inst in tasks:
            fut = executor.submit(_solve_instance_with_frrmab, list(key), inst)
            future_to_key[fut] = (key, inst)
        for fut in as_completed(future_to_key):
            key, inst = future_to_key[fut]
            try:
                val = fut.result()
            except Exception as e:
                logger.warning(f"滞留计数切换控制器运行失败，算子集合{key}在实例{inst}上: {e}")
                val = float('inf')
            runs.setdefault((key, inst), []).append(val)

    # 计算每个实例的平均makespan和相对误差
    result = {}
    skipped_instances = []
    for inst in train_instances:
        vals = runs.get((tuple(op_names), inst), [float('inf')])
        avg_makespan = mean(vals)
        
        # 从实例路径构建带前缀的键（如 'Small/VFR40_5_1_Gap.txt' 或 'Large/VFR100_20_1_Gap.txt'）
        # 实例路径格式: .../VRF_Instances/Small/VFR40_5_1_Gap.txt 或 .../VRF_Instances/Large/VFR100_20_1_Gap.txt
        instance_filename = os.path.basename(inst)
        # 获取父目录名（Small 或 Large）
        parent_dir = os.path.basename(os.path.dirname(inst))
        # 构建带前缀的键
        prefixed_key = f"{parent_dir}/{instance_filename}"
        
        # 从UB字典中查找对应实例的UB值
        if prefixed_key not in ub_dict:
            logger.warning(f"实例 {prefixed_key} 在UB字典中未找到，跳过")
            skipped_instances.append(prefixed_key)
            result[inst] = float('inf')
            continue
        
        ub = ub_dict[prefixed_key]
        
        # 计算相对误差: (avg_makespan - UB) / UB
        relative_error = (avg_makespan - ub) / ub
        result[inst] = relative_error
        
        logger.debug(f"实例 {prefixed_key}: 平均makespan={avg_makespan:.2f}, UB={ub}, 相对误差={relative_error:.6f} ({relative_error*100:.4f}%)")
    
    if skipped_instances:
        logger.warning(f"跳过了 {len(skipped_instances)} 个实例（未找到UB值）: {skipped_instances}")

    RESULTS_CACHE[tuple(op_names)] = result
    return result

def load_dataset():
    """
    从Large目录下选取指定规模的实例
    使用VFR100_20、VFR100_40、VFR100_60、VFR200_20、VFR200_40、VFR200_60这6种规模
    每种规模随机选取9个实例，总共6种规模 × 9个实例 = 54个训练实例
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    instances_root = os.path.join(current_dir, 'reevo-main', 'problems', 'pfsp_ig', 'VRF_Instances', 'Large')
    
    # 定义目标规模列表
    target_patterns = ['VFR100_20', 'VFR100_40', 'VFR100_60', 'VFR200_20', 'VFR200_40', 'VFR200_60']
    
    # 按"工件数_机器数"分组收集所有实例
    instances_by_pattern = {}
    
    # 扫描Large目录下的所有文件
    if os.path.exists(instances_root):
        for filename in os.listdir(instances_root):
            if filename.endswith('_Gap.txt') and filename.startswith('VFR'):
                # 解析文件名：VFR{工件数}_{机器数}_{编号}_Gap.txt
                parts = filename.replace('_Gap.txt', '').split('_')
                if len(parts) >= 3:
                    pattern = f"{parts[0]}_{parts[1]}"  # VFR100_20 格式
                    # 只处理目标规模
                    if pattern in target_patterns:
                        inst_path = os.path.join(instances_root, filename)
                        if pattern not in instances_by_pattern:
                            instances_by_pattern[pattern] = []
                        instances_by_pattern[pattern].append(inst_path)
    
    # 对每种规模随机选取9个实例
    train_instances = []
    selected_instances_by_pattern = {}
    
    for pattern in target_patterns:
        instances = instances_by_pattern.get(pattern, [])
        if len(instances) >= 9:
            # 随机选取9个
            selected = random.sample(instances, 9)
            train_instances.extend(selected)
            selected_instances_by_pattern[pattern] = selected
        elif len(instances) > 0:
            # 如果不足9个，则全部选取
            train_instances.extend(instances)
            selected_instances_by_pattern[pattern] = instances
            logger.warning(f"规模 {pattern} 只有 {len(instances)} 个实例，全部选取")
        else:
            logger.warning(f"规模 {pattern} 没有找到实例")
    
    # 记录日志
    logger.info(f"已加载 {len(train_instances)} 个训练实例（{len(selected_instances_by_pattern)}种规模，每种随机选取9个实例）")
    logger.info("使用的训练实例：")
    for pattern in sorted(selected_instances_by_pattern.keys()):
        selected = selected_instances_by_pattern[pattern]
        # 提取实例编号用于显示
        inst_names = []
        for inst_path in selected:
            filename = os.path.basename(inst_path)
            # 从 VFR100_20_1_Gap.txt 提取 VFR100_20_1
            inst_name = filename.replace('_Gap.txt', '')
            inst_names.append(inst_name)
        inst_names_str = ', '.join(sorted(inst_names))
        logger.info(f"  {pattern}: {inst_names_str}")
    
    return train_instances

# 删除 load_configuration() 函数，直接在代码中设置参数

def save_generation_results(generation, candidates, best_candidate, best_metric, current_op_pool, base_perf_map, cand_perf_maps):
    # 使用全局的_run_output_dir
    generation_dir = os.path.join(_run_output_dir, f'generation_{generation}')
    os.makedirs(generation_dir, exist_ok=True)

    # save candidate codes
    for i, alg in enumerate(candidates):
        alg_code_path = os.path.join(generation_dir, f'candidate_{i}_{alg.name}.py')
        with open(alg_code_path, 'w', encoding='utf-8') as f:
            f.write(alg.code)

    import json
    # base performance
    with open(os.path.join(generation_dir, 'base_performance.json'), 'w', encoding='utf-8') as f:
        json.dump({os.path.basename(k): v for k, v in base_perf_map.items()}, f, indent=2, ensure_ascii=False)
    # each candidate performance
    summary = {}
    for alg, perf in cand_perf_maps.items():
        summary[alg.name] = {os.path.basename(k): v for k, v in perf.items()}
    with open(os.path.join(generation_dir, 'candidates_performance.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # selection result
    selection_info = {
        'generation': generation,
        'current_op_pool': [*current_op_pool],
        'best_candidate': best_candidate.name if best_candidate else None,
        'metric_type': 'RelativeError',
        'best_relative_error': best_metric,
        'invalid_candidates_count': INVALID_CANDIDATES_COUNT.get(generation, 0),
        'total_candidates': len(candidates),
        'valid_candidates': len(candidates) - INVALID_CANDIDATES_COUNT.get(generation, 0)
    }
    with open(os.path.join(generation_dir, 'selection_result.json'), 'w', encoding='utf-8') as f:
        json.dump(selection_info, f, indent=2, ensure_ascii=False)
    logger.info(f"第{generation}代结果已保存到 {generation_dir}")

def average_perf(perf_map):
    """
    计算性能映射的平均值。
    现在输入的是相对误差，返回平均相对误差。
    """
    vals = list(perf_map.values())
    return float('inf') if not vals else sum(vals) / len(vals)

def main():
    # 在主函数开始时配置日志
    global _run_output_dir
    _run_output_dir = setup_logging()
    
    logger.info("启动自动滞留计数切换控制器-Reevo训练器（串行超启发式选择）...")
    logger.info("初始算子池包含初始动态破坏算子")

    # 直接设置配置参数
    num_candidates = 10  # 每轮生成10个候选算法
    max_workers = 300  # 最大并行工作进程数
    repeats = 20  # 每个算法在每个实例上运行20次
    force_regenerate = True  # 是否强制重新生成算法

    train_set = load_dataset()
    if not train_set:
        logger.error("未找到训练实例！")
        return

    # 使用初始算子作为算子池中的第一个算子进行初始化
    OP = ["initial_operator_gen0"]  # 带世代后缀的算子模块名称池，用于记录
    original_names_in_pool = ["initial_operator"]  # 不带_gen后缀的名称，求解器需要

    # 创建算子池目录并写入初始算子代码
    operator_pool_dir = os.path.join(_run_output_dir, 'operator_pool')
    os.makedirs(operator_pool_dir, exist_ok=True)
    
    # 定义初始算子的代码（将被写入到 initial_operator_gen0.py 文件中）
    initial_operator_code = '''def destroy_v2(sequence: list, processing_times: list) -> tuple[list, list]:
    from math import ceil

    # 计算给定工件序列的makespan
    def calculate_makespan(seq, proc_times):
        num_jobs = len(seq)
        num_machines = len(proc_times[0])
        completion_times = [[0] * num_machines for _ in range(num_jobs)]
        
        for i in range(num_jobs):
            job = seq[i]
            for m in range(num_machines):
                if i == 0 and m == 0:
                    completion_times[i][m] = proc_times[job][m]
                elif i == 0:
                    completion_times[i][m] = completion_times[i][m-1] + proc_times[job][m]
                elif m == 0:
                    completion_times[i][m] = completion_times[i-1][m] + proc_times[job][m]
                else:
                    completion_times[i][m] = max(completion_times[i-1][m], completion_times[i][m-1]) + proc_times[job][m]
        
        return max(completion_times[-1])

    # 计算当前makespan
    current_makespan = calculate_makespan(sequence, processing_times)

    # 计算每个工件对makespan的影响
    job_impact = []
    for i in range(len(sequence)):
        temp_sequence = sequence.copy()
        temp_sequence.pop(i)
        new_makespan = calculate_makespan(temp_sequence, processing_times)
        job_impact.append((sequence[i], (current_makespan - new_makespan)))

    # 按照影响大小排序
    job_impact.sort(key=lambda x: x[1], reverse=True)

    # 动态调整破坏数量，最多破坏4个工件
    d_max = 4
    d_min = 1
    d = min(max(d_min, ceil(current_makespan / (len(sequence) * 10))), d_max)

    # 优先移除对makespan影响较大的工件
    removed_jobs = [job for job, impact in job_impact[:d]]
    
    # 引入随机性以避免局部最优
    import random
    random.shuffle(removed_jobs)

    remaining_sequence = [job for job in sequence if job not in removed_jobs]

    return remaining_sequence, removed_jobs
'''
    
    initial_operator_file = os.path.join(operator_pool_dir, 'initial_operator_gen0.py')
    with open(initial_operator_file, 'w', encoding='utf-8') as f:
        f.write(initial_operator_code)
    logger.info(f"已创建初始算子文件: {initial_operator_file}")

    # 训练 MAX_OP_SIZE - 1 轮，每轮添加一个最优算子，最终算子池包含 MAX_OP_SIZE 个算子
    for generation in range(MAX_OP_SIZE - 1):
        logger.info(f"\n[第{generation + 1}代] 生成候选算法...")
        candidates = generate_algorithms(num_candidates, OP, generation, force_regenerate)
        if not candidates:
            logger.error("未生成候选算法！")
            break

        # 评估基础算子池（当前池上的滞留计数切换控制器）
        base_key = tuple(original_names_in_pool)
        if base_key not in RESULTS_CACHE:
            base_perf_map = evaluate_opset_on_instances(base_key, train_set, repeats, max_workers, generation)
        else:
            base_perf_map = RESULTS_CACHE[base_key]
        base_avg = average_perf(base_perf_map)
        logger.info(f"基础算子池 |OP|={len(base_key)} 平均相对误差: {base_avg:.6f} ({base_avg*100:.4f}%)")
        
        # 对于第一代，我们已经有了内置算子2，所以需要评估
        # 将每个候选算法添加到现有池时的改进

        # 评估每个附加到算子池的候选算法
        cand_perf_maps = {}
        invalid_before = len(INVALID_CANDIDATES)
        best_candidate = None
        best_avg = float('inf')

        for alg in candidates:
            # 通过将新算法添加到现有池来创建候选键
            cand_key = tuple(list(original_names_in_pool) + [alg.name])
            if cand_key not in RESULTS_CACHE:
                perf_map = evaluate_opset_on_instances(cand_key, train_set, repeats, max_workers, generation)
            else:
                perf_map = RESULTS_CACHE[cand_key]
            cand_perf_maps[alg] = perf_map

            cand_avg = average_perf(perf_map)
            # 计算性能改进（相对误差越小越好，所以改进 = base_avg - cand_avg）
            if base_avg != float('inf') and cand_avg != float('inf'):
                improvement = base_avg - cand_avg  # 相对误差的减少量
                improvement_percent = (improvement / base_avg) * 100 if base_avg > 0 else 0
                logger.info(f"候选算法 {alg.name} 平均相对误差={cand_avg:.6f} ({cand_avg*100:.4f}%) (改进 {improvement:.6f}, {improvement_percent:.2f}%)")
            else:
                logger.info(f"候选算法 {alg.name} 平均相对误差={cand_avg:.6f} (对比基础平均={base_avg:.6f})")

            if cand_avg < best_avg:
                best_avg = cand_avg
                best_candidate = alg

        INVALID_CANDIDATES_COUNT[generation] = max(0, len(INVALID_CANDIDATES) - invalid_before)
        if INVALID_CANDIDATES_COUNT[generation] > 0:
            logger.info(f"第{generation}代: 检测到 {INVALID_CANDIDATES_COUNT[generation]} 个新的无效算子集合")

        save_generation_results(generation, candidates, best_candidate, best_avg, original_names_in_pool, base_perf_map, cand_perf_maps)

        # 强制选择性能最好的候选算子，即使没有改进
        if best_candidate is None:
            # 如果没有有效的候选算子，选择第一个候选算子
            best_candidate = candidates[0] if candidates else None
            if best_candidate:
                best_avg = average_perf(cand_perf_maps[best_candidate])
                logger.warning(f"所有候选算子都无效，强制选择第一个候选算子 {best_candidate.name}")
            else:
                logger.error("没有可用的候选算子，停止运行。")
                break
        else:
            # 计算最终选择的候选算子的性能改进（相对误差越小越好）
            if base_avg != float('inf') and best_avg != float('inf'):
                final_improvement = base_avg - best_avg  # 相对误差的减少量
                final_improvement_percent = (final_improvement / base_avg) * 100 if base_avg > 0 else 0
                logger.info(f"已选择候选算法 {best_candidate.name} (平均相对误差={best_avg:.6f} ({best_avg*100:.4f}%), 改进 {final_improvement:.6f}, {final_improvement_percent:.2f}%)，新的 |OP|={len(OP)+1}")
            else:
                logger.info(f"已选择候选算法 {best_candidate.name} (平均相对误差={best_avg:.6f})，新的 |OP|={len(OP)+1}")

        OP.append(best_candidate.name)
        original_names_in_pool.append(best_candidate.name)
        
        # 新增：复制选中的算子到专门的算子池目录
        import shutil
        operator_pool_dir = os.path.join(_run_output_dir, 'operator_pool')
        os.makedirs(operator_pool_dir, exist_ok=True)
        
        # 修正：从 output_algorithms 读取基础算子文件
        import re
        match = re.match(r'(algorithm_\d+)_gen(\d+)', best_candidate.name)
        if not match:
            raise RuntimeError(f"候选算子名称不符合规范: {best_candidate.name}")
        base_name, gen_num = match.groups()
        src_file = os.path.join(os.path.dirname(__file__), 'reevo-main', 'output_algorithms', f'{base_name}.py')
        dst_file = os.path.join(operator_pool_dir, f'{base_name}_gen{gen_num}.py')
        shutil.copy2(src_file, dst_file)
        logger.info(f"已复制选中算子到算子池: {dst_file}")

    # 保存最终算子池和性能
    import json
    final_info = {
        'total_ops': len(OP),
        'op_names': OP,
        'original_module_names': original_names_in_pool,
        'initial_operator': 'initial_operator (Initial Destruction)',
        'generated_operators': [name for name in OP if not name.startswith('initial_operator')]
    }
    with open(os.path.join(_run_output_dir, 'final_op_pool.json'), 'w', encoding='utf-8') as f:
        json.dump(final_info, f, indent=2, ensure_ascii=False)

    # 保存最终算子池的最佳组合性能
    final_key = tuple(original_names_in_pool)
    if final_key not in RESULTS_CACHE:
        final_perf = evaluate_opset_on_instances(final_key, train_set, repeats, max_workers, generation=len(OP))
    else:
        final_perf = RESULTS_CACHE[final_key]
    with open(os.path.join(_run_output_dir, 'final_op_performance.json'), 'w', encoding='utf-8') as f:
        json.dump({os.path.basename(k): v for k, v in final_perf.items()}, f, indent=2, ensure_ascii=False)

    logger.info(f"\n最终算子池构建成功，包含 {len(OP)} 个算子。")
    final_avg_error = average_perf(final_perf)
    logger.info(f"最终平均相对误差: {final_avg_error:.6f} ({final_avg_error*100:.4f}%)")

if __name__ == "__main__":
    main()

