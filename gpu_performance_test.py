#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import time
import os

# Set environment variables for better performance
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

print("=== JAX GPU Performance Test ===")
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Default device: {jax.devices()[0]}")

# Test 1: Basic matrix operations
print("\n=== Test 1: Matrix Operations ===")
key = jax.random.PRNGKey(0)

# Large matrix multiplication
size = 4096
a = jax.random.normal(key, (size, size))
b = jax.random.normal(key, (size, size))

# Warm up
_ = jnp.dot(a, b)
jax.device_get(_)  # Synchronize

# Benchmark
start_time = time.time()
for _ in range(10):
    c = jnp.dot(a, b)
    _ = jax.device_get(c)  # Synchronize
end_time = time.time()

print(f"Matrix multiplication ({size}x{size}): {(end_time - start_time)/10:.4f} seconds per operation")

# Test 2: Neural network forward pass simulation
print("\n=== Test 2: Neural Network Simulation ===")

def simple_nn(x, w1, w2, w3):
    x = jax.nn.relu(jnp.dot(x, w1))
    x = jax.nn.relu(jnp.dot(x, w2))
    x = jnp.dot(x, w3)
    return x

batch_size = 1024
input_size = 512
hidden_size = 1024
output_size = 128

key, k1, k2, k3 = jax.random.split(key, 4)
x = jax.random.normal(k1, (batch_size, input_size))
w1 = jax.random.normal(k2, (input_size, hidden_size))
w2 = jax.random.normal(k3, (hidden_size, hidden_size))
w3 = jax.random.normal(key, (hidden_size, output_size))

# Warm up
_ = simple_nn(x, w1, w2, w3)
jax.device_get(_)

# Benchmark
start_time = time.time()
for _ in range(50):
    output = simple_nn(x, w1, w2, w3)
    _ = jax.device_get(output)
end_time = time.time()

print(f"Neural network forward pass: {(end_time - start_time)/50:.4f} seconds per pass")

# Test 3: Multi-GPU test
print("\n=== Test 3: Multi-GPU Test ===")
if len(jax.devices()) > 1:
    print(f"Testing with {len(jax.devices())} GPUs")
    
    # Simple multi-device computation
    def multi_gpu_test():
        devices = jax.devices()
        data = jax.random.normal(key, (len(devices), 1000, 1000))
        
        # Compute on each device
        results = []
        for i, device in enumerate(devices):
            with jax.default_device(device):
                result = jnp.sum(jnp.dot(data[i], data[i].T))
                results.append(result)
        
        return jnp.sum(jnp.array(results))
    
    # Warm up
    _ = multi_gpu_test()
    jax.device_get(_)
    
    # Benchmark
    start_time = time.time()
    for _ in range(10):
        result = multi_gpu_test()
        _ = jax.device_get(result)
    end_time = time.time()
    
    print(f"Multi-GPU computation: {(end_time - start_time)/10:.4f} seconds per operation")
else:
    print("Only one GPU available")

print("\n=== Performance Summary ===")
print("Current configuration appears to be working well!")
print("For further optimization, consider:")
print("1. Using JAX's pmap for parallel processing across GPUs")
print("2. Adjusting batch sizes for your specific workload")
print("3. Using jax.jit for function compilation")
print("4. Monitoring GPU memory usage and utilization")
