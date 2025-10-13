# WSRL Pipeline Scripts

This directory contains comprehensive pipeline scripts for running WSRL experiments across different D4RL dataset categories.

## Available Pipeline Scripts

### Individual Category Pipelines

1. **`run_locomotion_pipeline.sh`** - MuJoCo locomotion tasks (HalfCheetah, Hopper, Walker2D, Ant)
2. **`run_adroit_pipeline.sh`** - Adroit manipulation tasks (Pen, Hammer, Relocate, Door)
3. **`run_antmaze_pipeline.sh`** - AntMaze navigation tasks
4. **`run_kitchen_pipeline.sh`** - Kitchen manipulation tasks
5. **`run_maze2d_pipeline.sh`** - Maze2D navigation tasks
6. **`run_bullet_pipeline.sh`** - PyBullet physics tasks
7. **`run_flow_pipeline.sh`** - Flow traffic simulation tasks
8. **`run_carla_pipeline.sh`** - CARLA autonomous driving tasks
9. **`run_minigrid_pipeline.sh`** - MiniGrid tasks

### Master Scripts

10. **`run_all_pipelines.sh`** - Runs all pipelines in parallel across multiple GPUs
11. **`run_multi_env_pipeline.sh`** - Flexible script to run multiple environments within a category

## Usage

### Running Individual Pipelines

```bash
# Run a specific pipeline on GPU 0
bash experiments/scripts/pipelines/run_locomotion_pipeline.sh 0

# Run kitchen pipeline on GPU 1
bash experiments/scripts/pipelines/run_kitchen_pipeline.sh 1
```

### Running All Pipelines in Parallel

```bash
# Run all pipelines across multiple GPUs (0-8)
bash experiments/scripts/pipelines/run_all_pipelines.sh
```

### Running Multiple Environments in a Category

```bash
# Run all locomotion environments on GPU 0
bash experiments/scripts/pipelines/run_multi_env_pipeline.sh locomotion 0

# Run all adroit environments on GPU 1
bash experiments/scripts/pipelines/run_multi_env_pipeline.sh adroit 1

# Available categories: locomotion, adroit, antmaze, kitchen, maze2d, bullet, flow, carla, minigrid
```

## Pipeline Structure

Each pipeline follows a consistent 3-step process:

1. **CALQL Pretraining** - Offline pretraining with CALQL algorithm
2. **WSRL (SAC)** - Online fine-tuning with SAC
3. **WSRL (SAC-BC)** - Online fine-tuning with SAC-BC (behavior cloning)

## Environment Categories and Datasets

### Locomotion (MuJoCo)
- `halfcheetah-medium-v0`, `hopper-medium-v0`, `walker2d-medium-v0`, `ant-medium-v0`
- **Reward Scaling**: 1.0 scale, 0.0 bias
- **Training Steps**: 250K offline

### Adroit (Manipulation)
- `pen-cloned-v0`, `hammer-cloned-v0`, `relocate-cloned-v0`, `door-cloned-v0`
- **Reward Scaling**: 10.0 scale, 5.0 bias (binary sparse rewards)
- **Training Steps**: 20K offline

### AntMaze (Navigation)
- `antmaze-umaze-v2`, `antmaze-medium-play-v2`, `antmaze-large-play-v2`
- **Reward Scaling**: 10.0 scale, -5.0 bias
- **Training Steps**: 1M offline

### Kitchen (Manipulation)
- `kitchen-partial-v0`, `kitchen-complete-v0`, `kitchen-mixed-v0`
- **Reward Scaling**: 1.0 scale, -4.0 bias
- **Training Steps**: 250K offline

### Maze2D (2D Navigation)
- `maze2d-umaze-v1`, `maze2d-medium-v1`, `maze2d-large-v1`
- **Reward Scaling**: 1.0 scale, 0.0 bias
- **Training Steps**: 500K offline

### Bullet (PyBullet Physics)
- `bullet-halfcheetah-medium-v0`, `bullet-hopper-medium-v0`, `bullet-walker2d-medium-v0`, `bullet-ant-medium-v0`
- **Reward Scaling**: 1.0 scale, 0.0 bias
- **Training Steps**: 250K offline

### Flow (Traffic Simulation) - ⚠️ Requires Additional Setup
- **Note**: Flow environments require additional installation steps (SUMO simulator)
- **Fallback**: Using Maze2D environments as substitutes
- **Reward Scaling**: 1.0 scale, 0.0 bias
- **Training Steps**: 200K offline

### CARLA (Autonomous Driving) - ⚠️ Requires Additional Setup
- **Note**: CARLA environments require additional installation steps
- **Fallback**: Using Maze2D environments as substitutes
- **Reward Scaling**: 1.0 scale, 0.0 bias
- **Training Steps**: 300K offline

### MiniGrid (Grid World)
- `minigrid-fourrooms-v0`, `minigrid-fourrooms-random-v0`
- **Reward Scaling**: 1.0 scale, 0.0 bias
- **Training Steps**: 100K offline

## Configuration Files

The pipelines use configuration files from `experiments/configs/train_config.py`:

**Available Configurations:**
- `locomotion_cql` / `locomotion_wsrl` - For locomotion, maze2d, bullet, flow, carla, minigrid tasks
- `adroit_cql` / `adroit_wsrl` - For adroit manipulation tasks  
- `antmaze_cql` / `antmaze_wsrl` - For antmaze navigation tasks
- `kitchen_cql` / `kitchen_wsrl` - For kitchen manipulation tasks

**Configuration Mappings:**
- **Locomotion tasks**: Use `locomotion_cql` / `locomotion_wsrl`
- **Adroit tasks**: Use `adroit_cql` / `adroit_wsrl`  
- **AntMaze tasks**: Use `antmaze_cql` / `antmaze_wsrl`
- **Kitchen tasks**: Use `kitchen_cql` / `kitchen_wsrl`
- **Other tasks** (maze2d, bullet, flow, carla, minigrid): Use `locomotion_cql` / `locomotion_wsrl`

## Environment Variables

All scripts set up the following environment variables:

```bash
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export TF_ENABLE_ONEDNN_OPTS=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_ENABLE_X64=false
export JAX_TRACEBACK_FILTERING=off
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_triton_gemm_any=true"
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
```

## Output and Logging

- **Checkpoints**: Saved to `/media/nudt3090/XYQ/ZCX/WSRL/wsrl_log/`
- **Logs**: Training logs saved to `${SAVE_ROOT}/calql_${ENV_ID}_seed${SEED}.log`
- **Wandb**: Experiment tracking via Weights & Biases

## GPU Requirements

- **Individual pipelines**: 1 GPU per pipeline
- **All pipelines**: 9 GPUs (0-8) for parallel execution
- **Multi-env pipelines**: 1 GPU per category

## Troubleshooting

1. **CUDA out of memory**: Reduce `XLA_PYTHON_CLIENT_MEM_FRACTION` or use smaller batch sizes
2. **Dataset not found**: Ensure `D4RL_DATASET_DIR` points to correct dataset location
3. **MuJoCo errors**: Check `LD_LIBRARY_PATH` includes MuJoCo installation
4. **JAX compilation issues**: Set `TF_ENABLE_ONEDNN_OPTS=0` and `JAX_ENABLE_X64=false`
5. **Environment not found**: Run `python experiments/scripts/pipelines/test_available_envs.py` to check which environments are available
6. **Flow/CARLA errors**: These environments require additional setup. Use Maze2D environments as fallbacks.

## Example Commands

```bash
# Run single environment
bash experiments/scripts/pipelines/run_locomotion_pipeline.sh 0

# Run all environments in locomotion category
bash experiments/scripts/pipelines/run_multi_env_pipeline.sh locomotion 0

# Run all pipelines in parallel
bash experiments/scripts/pipelines/run_all_pipelines.sh

# Run specific category with custom GPU
bash experiments/scripts/pipelines/run_kitchen_pipeline.sh 2
```
