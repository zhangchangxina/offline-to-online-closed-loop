#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import time

print("Testing JAX setup...")
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Simple test computation
def test_computation():
    # Test basic JAX operations
    print("\nTesting basic JAX operations...")
    
    # Create some test data
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1000, 1000))
    y = jax.random.normal(key, (1000, 1000))
    
    # Test matrix multiplication
    start_time = time.time()
    z = jnp.dot(x, y)
    end_time = time.time()
    
    print(f"Matrix multiplication time: {end_time - start_time:.4f} seconds")
    print(f"Result shape: {z.shape}")
    
    # Test some basic operations
    a = jnp.array([[1, 2], [3, 4]])
    b = jnp.array([[5, 6], [7, 8]])
    c = jnp.dot(a, b)
    print(f"Small matrix multiplication result:\n{c}")
    
    # Test JIT compilation
    print("\nTesting JIT compilation...")
    
    @jax.jit
    def simple_function(x, y):
        return jnp.dot(x, y) + jnp.sin(x)
    
    start_time = time.time()
    result = simple_function(x, y)
    end_time = time.time()
    
    print(f"JIT compiled function time: {end_time - start_time:.4f} seconds")
    print(f"JIT result shape: {result.shape}")
    
    return True

if __name__ == "__main__":
    try:
        test_computation()
        print("\n✅ JAX setup is working!")
        print("Note: Currently using CPU. For GPU acceleration, you need CUDA-enabled jaxlib.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


