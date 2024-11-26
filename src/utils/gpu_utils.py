import logging
import tensorflow as tf
import subprocess
import os

logger = logging.getLogger(__name__)

def setup_gpu():
    """Configure GPU settings with proper CUDA paths"""
    try:
        # Set CUDA environment variables
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
        os.environ['CUDA_DIR'] = '/usr/local/cuda'

        # Configure memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set memory limit (leave some for system)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1536)]  # 1.5GB limit
            )

            # Disable TensorFlow's JIT compilation warning
            tf.config.optimizer.set_jit(False)

            logger.info("GPU setup completed successfully")
            return True

        return False
    except Exception as e:
        logger.error(f"GPU setup error: {e}")
        return False

def get_gpu_info():
    """Get GPU information safely"""
    try:
        if not tf.test.is_built_with_cuda():
            return {"available": False, "message": "TensorFlow not built with CUDA"}

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return {"available": False, "message": "No GPU devices found"}

        return {
            "available": True,
            "count": len(gpus),
            "device_name": tf.test.gpu_device_name(),
            "cuda_version": tf.sysconfig.get_build_info()["cuda_version"],
            "compute_capability": tf.test.compute_capability()
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