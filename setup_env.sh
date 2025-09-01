#!/bin/bash

echo "=== 开始自动安装 WSRL 环境 ==="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate wsrl

echo "1. 升级 pip..."
pip install --upgrade pip

echo "2. 安装基础依赖..."
pip install gym==0.23.1 numpy==1.26.4 distrax==0.1.2 ml_collections tqdm chex==0.1.82 optax==0.1.5 absl-py scipy==1.11.2 wandb einops imageio moviepy pre-commit overrides cython patchelf orbax-checkpoint==0.3.5 flax==0.7.5

echo "3. 安装 JAX (CPU 版本，避免网络问题)..."
pip install jax==0.4.13 jaxlib==0.4.13

echo "4. 安装其他依赖..."
pip install pybullet pybullet_envs mujoco-py

echo "5. 设置环境变量..."
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export WANDB_BASE_URL=https://api.bandw.top
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_PY_MUJOCO_LIB=$HOME/.mujoco/mujoco210/bin/libmujoco210.so
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONPATH=/media/nudt3090/XYQ/ZCX/WSRL/wsrl-main/D4RL:${PYTHONPATH}
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export CUDA_VISIBLE_DEVICES=0
export JAX_PLUGINS=none

echo "6. 测试 JAX 安装..."
python -c "import jax; print('JAX devices:', jax.devices()); print('JAX platform:', jax.default_backend())"

echo "7. 创建快速测试脚本..."
cat > quick_test.py << 'EOF'
#!/usr/bin/env python3
"""
快速测试脚本 - 大幅减少步数用于调试
"""

import os
import sys
sys.path.append('/media/nudt3090/XYQ/ZCX/WSRL/wsrl-main')

# 设置环境变量
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLUGINS'] = 'none'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

from finetune import main
from absl import flags

if __name__ == "__main__":
    # 覆盖默认参数
    flags.FLAGS.num_offline_steps = 1000  # 从1M降到1K
    flags.FLAGS.num_online_steps = 1000   # 从500K降到1K
    flags.FLAGS.batch_size = 256          # 从1024降到256
    flags.FLAGS.log_interval = 100        # 每100步记录一次
    flags.FLAGS.eval_interval = 1000000000  # 关闭评估
    flags.FLAGS.save_interval = 1000000000  # 关闭保存
    flags.FLAGS.debug = True              # 关闭wandb
    
    # 设置其他参数
    flags.FLAGS.env = "antmaze-large-diverse-v2"
    flags.FLAGS.agent = "sac"
    flags.FLAGS.utd = 1
    flags.FLAGS.reward_scale = 10.0
    flags.FLAGS.reward_bias = -5.0
    flags.FLAGS.config = "experiments/configs/train_config.py:antmaze_wsrl"
    
    main(None)
EOF

echo "8. 运行快速测试..."
python quick_test.py

echo "=== 安装完成 ==="
