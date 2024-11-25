import logging
import tensorflow as tf
import subprocess
import os

logger = logging.getLogger(__name__)

def get_gpu_info():
    """Get detailed GPU information"""
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if not gpu_devices:
            return {"available": False, "message": "No GPU devices found"}

        # Get memory info using nvidia-smi
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu',
             '--format=csv,noheader,nounits'],
            encoding='utf-8'
        ).strip().split(',')

        return {
            "available": True,
            "count": len(gpu_devices),
            "memory_used": int(result[0]),
            "memory_total": int(result[1]),
            "temperature": int(result[2]),
            "utilization": int(result[3]),
            "device_name": tf.test.gpu_device_name(),
            "cuda_version": tf.sysconfig.get_build_info()["cuda_version"]
        }
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        return {"available": False, "error": str(e)}

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

def setup_gpu():
    """Configure GPU settings for optimal performance with 4GB VRAM"""
    try:
        # Enable memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set memory limit (leave some for system)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]  # 3GB limit
            )

            # Enable mixed precision
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

            logger.info("GPU setup completed successfully")
            return True
        return False
    except Exception as e:
        logger.error(f"GPU setup error: {e}")
        return False

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