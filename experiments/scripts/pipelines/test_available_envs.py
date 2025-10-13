#!/usr/bin/env python3
"""
Test script to check which D4RL environments are actually available.
"""

import gym
import d4rl
import sys

# List of environments to test
ENVS_TO_TEST = [
    # Locomotion
    'halfcheetah-medium-v0', 'hopper-medium-v0', 'walker2d-medium-v0', 'ant-medium-v0',
    
    # Adroit
    'pen-cloned-v0', 'hammer-cloned-v0', 'relocate-cloned-v0', 'door-cloned-v0',
    
    # AntMaze
    'antmaze-umaze-v2', 'antmaze-medium-play-v2', 'antmaze-large-play-v2',
    
    # Kitchen
    'kitchen-partial-v0', 'kitchen-complete-v0', 'kitchen-mixed-v0',
    
    # Maze2D
    'maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1',
    
    # Bullet
    'bullet-halfcheetah-medium-v0', 'bullet-hopper-medium-v0', 'bullet-walker2d-medium-v0', 'bullet-ant-medium-v0',
    
    # MiniGrid
    'minigrid-fourrooms-v0', 'minigrid-fourrooms-random-v0',
    
    # Flow (may not be available)
    'flow-ring-random-v0', 'flow-ring-controller-v0',
    
    # CARLA (may not be available)
    'carla-lane-v0', 'carla-town-v0',
]

def test_env(env_name):
    """Test if an environment can be created."""
    try:
        env = gym.make(env_name)
        dataset = env.get_dataset()
        print(f"✅ {env_name} - Available (dataset size: {len(dataset['observations'])})")
        return True
    except Exception as e:
        print(f"❌ {env_name} - Not available: {str(e)}")
        return False

def main():
    print("Testing D4RL environment availability...")
    print("=" * 60)
    
    available_envs = []
    unavailable_envs = []
    
    for env_name in ENVS_TO_TEST:
        if test_env(env_name):
            available_envs.append(env_name)
        else:
            unavailable_envs.append(env_name)
    
    print("\n" + "=" * 60)
    print(f"Summary: {len(available_envs)} available, {len(unavailable_envs)} unavailable")
    
    if available_envs:
        print("\n✅ Available environments:")
        for env in available_envs:
            print(f"  - {env}")
    
    if unavailable_envs:
        print("\n❌ Unavailable environments:")
        for env in unavailable_envs:
            print(f"  - {env}")
    
    print("\nRecommended pipeline environments:")
    print("  - Locomotion: halfcheetah-medium-v0, hopper-medium-v0, walker2d-medium-v0, ant-medium-v0")
    print("  - Adroit: pen-cloned-v0, hammer-cloned-v0, relocate-cloned-v0, door-cloned-v0")
    print("  - AntMaze: antmaze-umaze-v2, antmaze-medium-play-v2, antmaze-large-play-v2")
    print("  - Kitchen: kitchen-partial-v0, kitchen-complete-v0, kitchen-mixed-v0")
    print("  - Maze2D: maze2d-umaze-v1, maze2d-medium-v1, maze2d-large-v1")
    print("  - Bullet: bullet-halfcheetah-medium-v0, bullet-hopper-medium-v0, bullet-walker2d-medium-v0, bullet-ant-medium-v0")
    print("  - MiniGrid: minigrid-fourrooms-v0, minigrid-fourrooms-random-v0")

if __name__ == "__main__":
    main()

