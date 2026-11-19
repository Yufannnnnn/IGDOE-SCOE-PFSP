import hydra
import logging 
import os
from pathlib import Path
import subprocess
from utils.utils import init_client, print_hyperlink


ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {print_hyperlink(workspace_dir)}")
    logging.info(f"Project Root: {print_hyperlink(ROOT_DIR)}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    client = init_client(cfg)
    # optional clients for operators (ReEvo)
    long_ref_llm = hydra.utils.instantiate(cfg.llm_long_ref) if cfg.get("llm_long_ref") else None
    short_ref_llm = hydra.utils.instantiate(cfg.llm_short_ref) if cfg.get("llm_short_ref") else None
    crossover_llm = hydra.utils.instantiate(cfg.llm_crossover) if cfg.get("llm_crossover") else None
    mutation_llm = hydra.utils.instantiate(cfg.llm_mutation) if cfg.get("llm_mutation") else None
    
    if cfg.algorithm == "reevo":
        from reevo import ReEvo as LHH
    else:
        raise NotImplementedError

    # Main algorithm
    lhh = LHH(cfg, ROOT_DIR, client, long_reflector_llm=long_ref_llm, short_reflector_llm=short_ref_llm, 
              crossover_llm=crossover_llm, mutation_llm=mutation_llm)
    best_codes, best_code_paths, best_objs = lhh.evolve()
    # 保存10个最优函数源码到 output_algorithms 目录，并记录obj
    output_dir = os.path.join(ROOT_DIR, "output_algorithms")
    os.makedirs(output_dir, exist_ok=True)
    for i, (code, code_path, obj) in enumerate(zip(best_codes, best_code_paths, best_objs)):
        save_path = os.path.join(output_dir, f"algorithm_{i}.py")
        with open(save_path, "w") as f:
            f.write(code)
        logging.info(f"Saved algorithm_{i}.py | code_path: {code_path} | obj: {obj}")
    logging.info(f"10 best codes saved to {output_dir}")

if __name__ == "__main__":
    main()