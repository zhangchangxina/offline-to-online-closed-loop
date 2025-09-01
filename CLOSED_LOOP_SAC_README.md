# SAC with Closed-Loop Update Mechanism

This implementation extends SAC with a closed-loop update mechanism that incorporates alignment loss and Lagrangian constraints for better training stability. This can be used independently or combined with warm-start strategies like WSRL.

## Architecture Overview

The closed-loop mechanism is designed as a **separate component** from warm-start strategies:

1. **WSRL Mechanism**: Collects small amounts of data before online training for warm-start
2. **Closed-Loop Mechanism**: Provides stable training updates during online fine-tuning

This separation allows for flexible combinations:
- Standard SAC + WSRL
- Closed-Loop SAC + WSRL  
- Closed-Loop SAC (standalone)
- Standard SAC (standalone)

## Available Method Combinations

### 🎯 **Recommended: Closed-Loop SAC + WSRL**
This is the most powerful combination, providing both warm-start benefits and stable training:

```bash
python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--project wsrl-closed-loop-combined \
--env antmaze-large-diverse-v2 \
--num_offline_steps 1_000_000 \
--warmup_steps 5000 \
--num_online_steps 500_000
```

**Benefits:**
- ✅ WSRL warm-start with offline data collection
- ✅ Closed-loop stable training with alignment constraints
- ✅ Enhanced monitoring and metrics
- ✅ Best performance in most scenarios

### 🔄 **Standard: Standard SAC + WSRL**
The original WSRL approach:

```bash
python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:antmaze_wsrl \
--project wsrl-standard \
--env antmaze-large-diverse-v2 \
--num_offline_steps 1_000_000 \
--warmup_steps 5000 \
--num_online_steps 500_000
```

**Benefits:**
- ✅ WSRL warm-start
- ✅ Standard SAC training
- ✅ Proven baseline

### 🚀 **Standalone: Closed-Loop SAC Only**
For scenarios where you don't need warm-start:

```bash
python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--project closed-loop-only \
--env antmaze-large-diverse-v2 \
--num_offline_steps 0 \
--num_online_steps 500_000
```

**Benefits:**
- ✅ Stable training with alignment constraints
- ✅ No offline data requirements
- ✅ Good for online-only scenarios

## Quick Start Guide

### 1. **For Best Performance (Recommended)**
Use the combined approach:

```bash
bash train_wsrl_closed_loop_combined.sh
```

### 2. **For Method Comparison**
Compare different approaches:

```bash
bash compare_methods.sh
```

### 3. **For Individual Testing**
Test specific combinations:

```bash
# Closed-Loop SAC + WSRL
python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--num_offline_steps 1_000_000 \
--warmup_steps 5000 \
--env antmaze-large-diverse-v2

# Standard SAC + WSRL
python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:antmaze_wsrl \
--num_offline_steps 1_000_000 \
--warmup_steps 5000 \
--env antmaze-large-diverse-v2
```

## Key Components

### 1. Alignment Loss
The alignment loss implements the constraint `E[q_delta^2] <= c` where:
- `q_delta = r + γQ(s',a') - Q(s,a)` is the temporal difference error
- `c` is the constraint threshold (default: 0.1)

This prevents the Q-function from deviating too far from the Bellman equation.

### 2. Lagrangian Objective
The policy loss combines three terms:
```python
policy_loss = q_term + lam_align * align_loss + entropy_term
```

Where:
- `align_loss = E[q_delta^2]`
- `q_term = (-q_new).mean()` (maximizes Q-values)
- `entropy_term = (temperature * log_probs).mean()` (entropy regularization)

### 3. Lambda Schedule
`lam_align` is determined by `lambda_schedule`:

```python
if schedule == "fixed":
    lam_align = lam_align_init
elif schedule == "linear":
    lam_align = lam_align_init * (1 - progress)
elif schedule == "adaptive":
    lam_align = align_lagrange  # learned multiplier
```

## Usage

### 1. Basic Usage

```python
from wsrl.agents import ClosedLoopSACAgent

# Create agent with closed-loop updates
agent = ClosedLoopSACAgent.create(
    rng=rng,
    observations=observations,
    actions=actions,
    encoder_def=encoder_def,
    # Closed-loop specific parameters
    align_constraint=0.1,  # Constraint threshold
    lam_align=1.0,         # Alignment loss weight (init)
    # use lambda_schedule to control lam_align
)
```

### 2. Training Script

```bash
python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--project sac-closed-loop \
--env antmaze-large-diverse-v2 \
--num_offline_steps 1_000_000 \
--num_online_steps 500_000 \
--seed 0
```

### 3. Available Configurations

- `antmaze_closed_loop_sac`: For antmaze environments
- `adroit_closed_loop_sac`: For adroit environments  
- `kitchen_closed_loop_sac`: For kitchen environments
- `locomotion_closed_loop_sac`: For locomotion environments

## Configuration Parameters

### Core Parameters
- `align_constraint` (float): Constraint threshold for alignment loss (default: 0.1)
- `lam_align` (float): Weight for alignment loss (default: 1.0)

### Schedule Parameters
- `lambda_schedule` (str): One of {"fixed", "linear", "adaptive"}
- `lam_eff_linear_steps` (int): Steps for linear decay (default: 100000)

## Monitoring

The agent provides additional metrics for monitoring the closed-loop mechanism:

```python
info = {
    "align_loss": align_loss,                    # Alignment loss value
    "constraint_violation": constraint_violation, # How much constraint is violated
    "q_delta_mean": q_delta_mean,                # Mean of Q-delta
    "q_delta_std": q_delta_std,                  # Std of Q-delta
    "lam_align": lam_align,                      # Effective lambda for alignment
    "q_term": q_term,                            # Q-term value
    "entropy_term": entropy_term,                # Entropy term value
}
```

## Example Scripts

1. **Basic Example**: `example_closed_loop_sac.py` - Demonstrates agent creation and single update
2. **Training Script**: `train_closed_loop_sac.sh` - Complete training examples for different environments
3. **Combined Training**: `train_wsrl_closed_loop_combined.sh` - WSRL + Closed-Loop SAC combination
4. **Method Comparison**: `compare_methods.sh` - Compare different method combinations

## Comparison with Standard SAC

| Feature | Standard SAC | Closed-Loop SAC |
|---------|--------------|-----------------|
| Q-function updates | Standard | Constrained by alignment loss |
| Policy optimization | Standard | Lagrangian objective |
| Stability | Basic | Enhanced with constraints |
| Monitoring | Standard metrics | + Alignment metrics |

## Combining with WSRL

The closed-loop mechanism can be combined with WSRL for enhanced performance:

1. **Offline Phase**: Collect small amounts of data for warm-start
2. **Online Phase**: Use ClosedLoopSACAgent for stable fine-tuning

```bash
# Example: WSRL + Closed-Loop SAC
python3 finetune.py \
--agent closed_loop_sac \
--config experiments/configs/train_config.py:antmaze_closed_loop_sac \
--num_offline_steps 1_000_000 \
--warmup_steps 5_000 \
--num_online_steps 500_000
```

## Tips for Tuning

1. **Start with default values**: `align_constraint=0.1`, `lam_align=1.0`
2. **Adjust constraint**: Lower `align_constraint` for more conservative updates
3. **Use linear schedule**: choose `lambda_schedule="linear"` for gradual decay
5. **Monitor metrics**: Watch `constraint_violation` and `q_delta_std` for stability

## Implementation Details

The closed-loop mechanism is implemented in `wsrl/agents/closed_loop_sac.py` and extends the base `SACAgent` class. Key methods:

- `_compute_align_loss()`: Computes alignment loss and Q-delta
- `policy_loss_fn()`: Enhanced policy loss with Lagrangian objective
- `create()`: Factory method with closed-loop parameters

The implementation is fully JAX-compatible and supports all existing SAC features including ensemble critics, high UTD ratios, and multi-device training.

## Architecture Benefits

1. **Modularity**: Closed-loop mechanism is independent of warm-start strategies
2. **Flexibility**: Can be used with or without WSRL
3. **Compatibility**: Works with existing SAC infrastructure
4. **Extensibility**: Easy to combine with other algorithms

