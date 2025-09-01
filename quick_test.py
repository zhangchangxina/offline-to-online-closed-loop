#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import time
import os

# Set environment variables to disable GPU warnings
os.environ['JAX_PLUGINS'] = 'none'

print("Quick JAX Test")
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Simple test
def test_basic():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 10))
    y = jax.random.normal(key, (100, 1))
    
    # Simple linear model
    w = jax.random.normal(key, (10, 1))
    b = jax.random.normal(key, (1,))
    
    def predict(params, x):
        w, b = params
        return jnp.dot(x, w) + b
    
    def loss(params, x, y):
        pred = predict(params, x)
        return jnp.mean((pred - y) ** 2)
    
    # Test forward pass
    start_time = time.time()
    params = (w, b)
    loss_val = loss(params, x, y)
    end_time = time.time()
    
    print(f"Loss: {loss_val:.4f}")
    print(f"Time: {end_time - start_time:.4f} seconds")
    
    # Test gradient
    grad_fn = jax.grad(loss)
    grads = grad_fn(params, x, y)
    print(f"Gradients computed successfully")
    
    return True

if __name__ == "__main__":
    try:
        test_basic()
        print("✅ Basic JAX functionality works!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
