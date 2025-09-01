#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import flax.linen as nn
import time

print("Testing JAX and Flax setup...")
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Simple test computation
def test_computation():
    # Create a simple neural network
    class SimpleNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(128)(x)
            x = jax.nn.relu(x)
            x = nn.Dense(64)(x)
            x = jax.nn.relu(x)
            x = nn.Dense(1)(x)
            return x
    
    # Initialize the model
    model = SimpleNet()
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 10))
    
    # Initialize parameters
    params = model.init(key, x)
    
    # Test forward pass
    start_time = time.time()
    output = model.apply(params, x)
    end_time = time.time()
    
    print(f"Model output shape: {output.shape}")
    print(f"Forward pass time: {end_time - start_time:.4f} seconds")
    
    # Test some basic JAX operations
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5, 6], [7, 8]])
    c = jnp.dot(a, b)
    print(f"Matrix multiplication result:\n{c}")
    
    return True

if __name__ == "__main__":
    try:
        test_computation()
        print("✅ JAX and Flax setup is working!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


