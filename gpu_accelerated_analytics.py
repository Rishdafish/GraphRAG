import time

# At the top of the file
HAS_GPU = False
HAS_NVML = False

# Simplified GPU performance tracker for CPU-only systems
class GPUPerformanceTracker:
    def __init__(self):
        self.has_gpu = False
        self.metrics = []
    
    def start_tracking(self, operation_name):
        self.operation_name = operation_name
        self.start_time = time.time()
    
    def stop_tracking(self):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        
        result = {
            "operation": self.operation_name,
            "execution_time": self.execution_time,
            "gpu_enabled": False,
            "gpu_metrics": None
        }
        
        self.metrics.append(result)
        return result
    
    # ... rest of the methods with simple fallbacks