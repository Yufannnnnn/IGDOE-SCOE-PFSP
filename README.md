# SCOE: Stagewise Cooperative Operator Ensemble Evolution

Welcome to this repository.

This repository is centered on **SCOE (Stagewise Cooperative Operator Ensemble Evolution)**. SCOE is an offline evolutionary framework that uses large language models to automatically design **destruction operator ensemble**. At present, the repository contains two main components:

1. `industrial_data/`: a **new industrial benchmark set** constructed from real production logs and processing requirements of an instrument-manufacturing workshop in Shanghai City.
2. `scoe/`: the core implementation of the SCOE framework, together with its current instantiation on the **PFSP** task.

This repository also preserves the third-party project `scoe/reevo-main/` as the foundational LLM-driven heuristic generation module. Our work is developed on top of ReEvo, but the focus is not merely improving heuristics for a single problem. Instead, it targets the more general problem of automatic operator ensemble design. The ReEvo project page is available at [ai4co.github.io/reevo](https://ai4co.github.io/reevo/).

## Table of Contents

* 1\. [Introduction](#Introduction)
* 2\. [Core Ideas](#Highlights)
* 3\. [Repository Structure](#RepositoryStructure)
* 4\. [Usage](#Usage)
  * 4.1. [Dependencies](#Dependency)
  * 4.2. [Industrial Benchmark](#IndustrialBenchmark)
  * 4.3. [Running the PFSP Instantiation of SCOE](#RunLLMAHD)
  * 4.4. [Outputs](#Outputs)
  * 4.5. [More Configurations](#MoreConfigs)
* 5\. [Citation](#Citation)
* 6\. [Acknowledgments and License](#Acknowledgments)

## 1. <a name='Introduction'></a>Introduction

SCOE (Stagewise Cooperative Operator Ensemble Evolution) is an offline evolutionary framework that uses large language models to automatically design cooperative heuristic operator sets. Its central goal is not to generate a single operator for a specific task, but to **construct a complementary, cooperative, and robust operator ensemble (DOE) for an existing algorithmic framework**.

More specifically, this repository serves two purposes:

- to provide an industrial benchmark derived from real production logs and processing requirements;
- to provide a reproducible framework for studying how SCOE can automatically design, evaluate, and expand cooperative operator sets.

Within this repository, `industrial_data/` represents the data side of the project, while `scoe/` implements the methodological side. The current repository uses **PFSP** as the instantiation and validation task, but SCOE itself is positioned as a **general framework for automatic operator-set design**, rather than a method limited to PFSP.

## 2. <a name='Highlights'></a>Core Ideas

The core logic of SCOE can be summarized in four aspects:

- **Incremental Construction Strategy**: the overall design problem is decomposed into stagewise generation steps, where the LLM is guided to generate only one best candidate operator at a time, and each new operator is expected to compensate for the capability gaps of the current operator pool.
- **State-Awareness Mechanism**: key features of the current operator pool, such as destruction scale, strategy type, and behavioral pattern, are extracted and fed back to the LLM, enabling it to perceive unexplored regions of the search space and generate complementary rather than redundant operator logic.
- **cooperative evaluation mechanism**: the value of a candidate operator is not measured by its standalone performance, but by its marginal contribution after being inserted into the full algorithmic framework.
- **Adaptive ensemble construction**: the framework ultimately aims to build a robust operator ensemble (DOE), while allowing the effective ensemble size to be determined by actual performance gains.

In this repository, the framework is instantiated on the PFSP task and coupled with the `IG` evaluator to form an executable offline operator-ensemble design pipeline.

In addition, the repository contains the following supporting components:

- **Real industrial benchmark**: the instances in `industrial_data/` come from actual production logs and processing requirements of an instrument-manufacturing workshop in Shanghai City, rather than being purely synthetic.
- **PFSP instantiation**: the `scoe/` directory provides a complete implementation of SCOE on PFSP, including operator generation, collaborative evaluation, and ensemble construction.
- **Implementation built on ReEvo**: the repository preserves and adapts the upstream `ReEvo` code to support LLM-based heuristic generation and reflective evolutionary search. See the upstream introduction at [ReEvo Project Page](https://ai4co.github.io/reevo/).

## 3. <a name='RepositoryStructure'></a>Repository Structure

The most important directories in this repository are:

- `README.md`: this document
- `LICENSE`: repository license
- `industrial_data/`: real industrial benchmark set
- `scoe/`: core code of the SCOE framework and its PFSP instantiation

More specifically:

- `industrial_data/` currently contains 12 real industrial instances, covering multiple job-machine scales such as `80x20`, `100x20`, `100x40`, `200x20`, `200x40`, and `300x20`.
- `scoe/main.py` is the main entry script for the current PFSP instantiation pipeline.
- `scoe/src/ig.py` implements the PFSP-IG solver logic and instance reader used in the current experiments.
- `scoe/src/connect.py` implements the connection, collaborative evaluation, and control logic between candidate operators and the full solving framework.
- `scoe/reevo-main/` contains the preserved third-party ReEvo framework and the related adaptations used to support LLM-based candidate operator generation. See the original project at [ReEvo Project Page](https://ai4co.github.io/reevo/).

## 4. <a name='Usage'></a>Usage

- If you want to inspect the real industrial instances directly, see `industrial_data/`.
- If you want to run the current PFSP instantiation of the method, use the workflow under `scoe/`.
- If you want to reproduce the experiments, first complete dependency installation and LLM API configuration.

#### 4.1. <a name='Dependency'></a>Dependencies

Python `>=3.11` is recommended.

Most runtime dependencies are defined in `scoe/reevo-main/pyproject.toml`. In addition, the main workflow depends on `pandas` and `openpyxl` for reading experimental data and Excel upper-bound files.

One workable setup is:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e ./scoe/reevo-main
pip install pandas openpyxl
```

If you prefer `uv`, you can also use the `pyproject.toml` and `uv.lock` files under `scoe/reevo-main/` to synchronize dependencies. That part of the framework is inherited from ReEvo; see [https://ai4co.github.io/reevo/](https://ai4co.github.io/reevo/).

#### 4.1.1. Upstream Framework Background

The LLM-based generation module in this work is built on ReEvo. ReEvo is a **Language Hyper-Heuristics** framework for combinatorial optimization that combines evolutionary search with LLM reflections to automate heuristic design and iterative improvement. For the original design philosophy, supported problem types, and paper information, please visit [https://ai4co.github.io/reevo/](https://ai4co.github.io/reevo/).

#### 4.2. <a name='IndustrialBenchmark'></a>Industrial Benchmark

`industrial_data/` is an important part of this repository and contains a new benchmark set constructed from:

> actual production logs and processing requirements of an instrument-manufacturing workshop in Shanghai City

These instances describe PFSP scheduling problems in a real workshop environment and serve as an important validation scenario for the current instantiation of the SCOE framework.

The instance files follow the naming pattern:

```text
RealPFSP_<jobs>_<machines>_<index>.txt
```

For example:

- `RealPFSP_80_20_1.txt`
- `RealPFSP_100_40_2.txt`
- `RealPFSP_300_20_1.txt`

In each instance file, the first line specifies `number_of_jobs number_of_machines`, and the remaining lines describe the processing-time information for each job.

#### 4.3. <a name='RunLLMAHD'></a>Running the PFSP Instantiation of SCOE

Run the main workflow from the repository root:

```bash
python scoe/main.py
```

The workflow automatically performs the following steps:

- initialize the run directory and logs;
- construct the initial operator;
- call `scoe/reevo-main/reevo_main.py` to generate candidate operators;
- insert candidate operators into the full PFSP-IG framework for collaborative effectiveness evaluation;
- select the best candidate according to the marginal performance gain after insertion;
- expand the operator ensemble stage by stage, and output the results of each generation together with the final DOE.

If LLM API access is required, configure the environment variables before running, for example:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

The relevant configuration files are located at:

- `scoe/reevo-main/cfg/config.yaml`
- `scoe/reevo-main/cfg/llm_client/openai.yaml`

Do not place real API keys directly into repository files.

#### 4.4. <a name='Outputs'></a>Outputs

After running `python scoe/main.py`, outputs are written by default to:

```text
scoe/output/run_YYYYMMDD_HHMMSS/
```

This directory typically contains:

- `logs/`: runtime logs
- `operator_pool/`: the operator pool retained during the current run
- `generation_k/`: candidate code snapshots and evaluation results for generation `k`
- `final_op_pool.json`: information about the final operator ensemble
- `final_op_performance.json`: performance results of the final operator ensemble on the training set

#### 4.5. <a name='MoreConfigs'></a>More Configurations

The current `scoe/main.py` includes several default experimental settings, such as:

- final operator ensemble size
- number of candidate operators per generation
- maximum number of parallel evaluation workers
- number of repeated evaluations per instance

ReEvo-related search parameters are located in:

- `scoe/reevo-main/cfg/config.yaml`

If needed, these settings can be further reorganized into a more standard command-line or experiment-configuration interface.

## 5. <a name='Acknowledgments'></a>Acknowledgments and License

This repository preserves a copy of the third-party open-source project **ReEvo** under `scoe/reevo-main/`:

- **Upstream**: ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution
- **Project Page**: [https://ai4co.github.io/reevo/](https://ai4co.github.io/reevo/)
- **License**: MIT

For the remaining parts of this repository, see the top-level `LICENSE`.

If you redistribute this repository while keeping the `scoe/reevo-main/` directory, please also retain its original license and copyright notice.