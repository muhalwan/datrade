# verify_gpu.py
import tensorflow as tf
import torch

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available (TensorFlow):", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA Available:", torch.cuda.is_available())
print("PyTorch CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("PyTorch CUDA Device Name:", torch.cuda.get_device_name(0))
