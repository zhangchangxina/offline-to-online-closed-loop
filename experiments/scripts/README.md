# Experiment Scripts

This directory contains organized scripts for running experiments on different environments.

## Directory Structure

```
experiments/scripts/
├── adroit/
│   ├── launch_wsrl_finetune.sh          # Standard WSRL (SAC + WSRL)
│   ├── launch_closed_loop_sac_finetune.sh # Closed-Loop SAC + WSRL
│   └── launch_comparison.sh             # Compare both methods
├── antmaze/
│   ├── launch_wsrl_finetune.sh          # Standard WSRL (SAC + WSRL)
│   ├── launch_closed_loop_sac_finetune.sh # Closed-Loop SAC + WSRL
│   └── launch_comparison.sh             # Compare both methods
├── locomotion/
│   ├── launch_wsrl_finetune.sh          # Standard WSRL (SAC + WSRL)
│   ├── launch_closed_loop_sac_finetune.sh # Closed-Loop SAC + WSRL
│   └── launch_comparison.sh             # Compare both methods
└── kitchen/
    ├── launch_wsrl_finetune.sh          # Standard WSRL (SAC + WSRL)
    ├── launch_closed_loop_sac_finetune.sh # Closed-Loop SAC + WSRL
    └── launch_comparison.sh             # Compare both methods
```

## Available Methods

### 1. **Standard WSRL** (Original)
- **Agent**: `sac`
- **Config**: `*_wsrl`
- **Description**: Standard SAC with WSRL warm-start mechanism

### 2. **Closed-Loop SAC + WSRL** (Recommended)
- **Agent**: `closed_loop_sac`
- **Config**: `*_closed_loop_sac`
- **Description**: SAC with closed-loop updates + WSRL warm-start

## Usage

### Individual Experiments

#### Adroit Environments
```bash
# Standard WSRL
cd experiments/scripts/adroit
bash launch_wsrl_finetune.sh

# Closed-Loop SAC + WSRL
bash launch_closed_loop_sac_finetune.sh

# Available environments: pen-binary-v0, door-binary-v0, relocate-binary-v0
```

#### Antmaze Environments
```bash
# Standard WSRL
cd experiments/scripts/antmaze
bash launch_wsrl_finetune.sh

# Closed-Loop SAC + WSRL
bash launch_closed_loop_sac_finetune.sh

# Available environments: antmaze-large-diverse-v2, antmaze-large-play-v2, antmaze-medium-diverse-v2, antmaze-medium-play-v2
```

#### Locomotion Environments
```bash
# Standard WSRL
cd experiments/scripts/locomotion
bash launch_wsrl_finetune.sh

# Closed-Loop SAC + WSRL
bash launch_closed_loop_sac_finetune.sh

# Available environments: halfcheetah-medium-replay-v2, halfcheetah-medium-v2, hopper-medium-replay-v2, hopper-medium-v2, walker2d-medium-replay-v2, walker2d-medium-v2
```

#### Kitchen Environments
```bash
# Standard WSRL
cd experiments/scripts/kitchen
bash launch_wsrl_finetune.sh

# Closed-Loop SAC + WSRL
bash launch_closed_loop_sac_finetune.sh

# Available environments: kitchen-partial-v0, kitchen-complete-v0, kitchen-mixed-v0
```

### Comparison Experiments

Run comparison experiments to compare WSRL vs Closed-Loop SAC + WSRL:

```bash
# Adroit comparison
cd experiments/scripts/adroit
bash launch_comparison.sh pen-binary-v0

# Antmaze comparison
cd experiments/scripts/antmaze
bash launch_comparison.sh antmaze-large-diverse-v2

# Locomotion comparison
cd experiments/scripts/locomotion
bash launch_comparison.sh halfcheetah-medium-replay-v2

# Kitchen comparison
cd experiments/scripts/kitchen
bash launch_comparison.sh kitchen-partial-v0
```

## Environment-Specific Settings

### Adroit
- **Offline Steps**: 20,000
- **Reward Scale**: 10.0
- **Reward Bias**: 5.0
- **Default Environment**: pen-binary-v0

### Antmaze
- **Offline Steps**: 1,000,000
- **Reward Scale**: 10.0
- **Reward Bias**: -5.0
- **Default Environment**: antmaze-large-diverse-v2

### Locomotion
- **Offline Steps**: 250,000
- **Reward Scale**: 1.0
- **Reward Bias**: 0.0
- **Default Environment**: halfcheetah-medium-replay-v2

### Kitchen
- **Offline Steps**: 250,000
- **Reward Scale**: 1.0
- **Reward Bias**: -4.0
- **Default Environment**: kitchen-partial-v0

## Common Parameters

All scripts use the following common parameters:
- **UTD**: 4 (update-to-data ratio)
- **Batch Size**: 1024
- **Warmup Steps**: 5000
- **Online Sampling Method**: mixed
- **Offline Data Ratio**: 0.0

## WandB Projects

Each environment has its own WandB project:
- **Adroit**: `method-section` (individual), `adroit-comparison` (comparison)
- **Antmaze**: `method-section` (individual), `antmaze-comparison` (comparison)
- **Locomotion**: `method-section` (individual), `locomotion-comparison` (comparison)
- **Kitchen**: `kitchen-finetune` (individual), `kitchen-comparison` (comparison)

## Customization

To customize experiments, you can:

1. **Change Environment**: Pass environment name as argument to comparison scripts
2. **Modify Parameters**: Edit the script files directly
3. **Add Arguments**: Use `$@` to pass additional arguments to finetune.py

Example:
```bash
# Run with custom parameters
bash launch_closed_loop_sac_finetune.sh --num_online_steps 1000000 --seed 42
```

## Expected Results

### Performance Comparison
- **Closed-Loop SAC + WSRL** should provide better stability during online fine-tuning
- **Standard WSRL** serves as the baseline for comparison
- Monitor alignment loss and constraint violation metrics for closed-loop performance

### Key Metrics to Monitor
- **Standard Metrics**: Actor loss, critic loss, entropy, temperature
- **Closed-Loop Metrics**: Alignment loss, constraint violation, Q-delta statistics
- **Performance Metrics**: Episode returns, success rates (environment-dependent)
