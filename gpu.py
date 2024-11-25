import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

import numpy as np
print(f"NumPy version: {np.__version__}")

import pandas as pd
print(f"Pandas version: {pd.__version__}")

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"\nNum GPUs Available: {len(gpus)}")
print(f"GPU Devices: {gpus}")
print(f"CUDA Available: {tf.test.is_built_with_cuda()}")

if gpus:
    try:
        # Try a simple GPU operation
        with tf.device('/GPU:0'):
            print("\nTesting GPU computation...")
            x = tf.random.normal([1000, 1000])
            y = tf.random.normal([1000, 1000])
            z = tf.matmul(x, y)
            print("GPU computation successful!")
    except Exception as e:
        print(f"\nGPU computation failed: {e}")