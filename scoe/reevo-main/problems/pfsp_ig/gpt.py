def destroy_v2(sequence: list, processing_times: list) -> tuple[list, list]:
    """
    随机移除2个工件
    输入：
        sequence: 当前工件序列
        processing_times: 工件在各机器上的处理时间
    输出：
        remaining_sequence: 破坏后剩余工件序列
        removed_jobs: 被移除的工件列表
    """
    import random
    d_max = 2
    d = min(d_max, len(sequence))
    positions = random.sample(range(len(sequence)), d)
    removed_jobs = [sequence[p] for p in sorted(positions, reverse=True)]
    remaining_sequence = sequence.copy()
    for p in sorted(positions, reverse=True):
        remaining_sequence.pop(p)
    return remaining_sequence, removed_jobs 
