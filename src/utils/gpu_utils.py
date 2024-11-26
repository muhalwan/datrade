import logging
import tensorflow as tf
import subprocess
import os
import torch

logger = logging.getLogger(__name__)

def get_gpu_info():
    """Get GPU information from both TensorFlow and PyTorch"""
    try:
        gpu_info = {
            "available": False,
            "tensorflow_gpus": [],
            "pytorch_cuda": False,
            "memory_info": None
        }

        # Check TensorFlow GPUs
        tf_gpus = tf.config.list_physical_devices('GPU')
        if tf_gpus:
            gpu_info["available"] = True
            gpu_info["tensorflow_gpus"] = [{
                "name": gpu.name,
                "device_type": gpu.device_type,
            } for gpu in tf_gpus]

            # Get memory info
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            if memory_info:
                gpu_info["memory_info"] = {
                    "peak": memory_info['peak'] / (1024 * 1024),  # Convert to MB
                    "current": memory_info['current'] / (1024 * 1024)
                }

        # Check PyTorch CUDA
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["pytorch_cuda"] = True
            gpu_info["cuda_device"] = torch.cuda.get_device_name(0)
            gpu_info["cuda_version"] = torch.version.cuda

        return gpu_info

    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return {"available": False, "error": str(e)}

def setup_gpu():
    """Configure GPU settings for both TensorFlow and PyTorch"""
    try:
        # Do TensorFlow GPU setup first, before any GPU operations
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Set memory limit only for the first GPU
                memory_limit = int(1024 * 3)  # 3GB limit
                logical_gpus = [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                tf.config.set_logical_device_configuration(gpus[0], logical_gpus)

            except RuntimeError as e:
                logger.warning(f"GPU memory configuration must be set before GPUs are initialized: {e}")

        # PyTorch GPU setup
        if torch.cuda.is_available():
            # Set the current GPU device
            torch.cuda.set_device(0)
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.benchmark = True
            # For reproducibility
            torch.backends.cudnn.deterministic = True

        return True

    except Exception as e:
        logger.error(f"GPU setup error: {e}")
        return False


def get_gpu_memory_info():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used, total = map(int, result.strip().split(','))
        return {
            'total_memory': total,
            'used_memory': used
        }
    except:
        return None

def monitor_gpu_usage():
    """Monitor current GPU usage"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        ).strip().split(',')

        return {
            "memory_used_mb": int(result[0]),
            "memory_total_mb": int(result[1]),
            "temperature_c": int(result[2]),
            "utilization_percent": int(result[3])
        }
    except Exception as e:
        logger.error(f"Error monitoring GPU: {e}")
        return None