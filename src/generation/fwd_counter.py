import time
from contextlib import contextmanager

class ForwardHookCounter:
    '''Uses PyTorch hooks to count forward passes and track wall-clock time.'''
    
    def __init__(self, model):
        self.model = model
        self.nfe = 0
        self.elapsed = 0.0
        self.hook_handle = None
        self._start_time = None
        
    def _forward_hook(self, module, input, output):
        self.nfe += 1
        return output
        
    def start(self):
        self.nfe = 0
        self._start_time = time.time()
        self.hook_handle = self.model.register_forward_hook(self._forward_hook)
        
    def stop(self):
        self.elapsed = time.time() - self._start_time if self._start_time else 0.0
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
        
    def reset(self):
        self.nfe = 0
        self.elapsed = 0.0
        self._start_time = None

    @property
    def avg_forward_time(self):
        return self.elapsed / self.nfe if self.nfe > 0 else 0.0
        
    @contextmanager
    def count(self):
        '''Context manager that tracks both NFE and wall-clock time.'''
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def __str__(self):
        return f"time: {self.elapsed:.4f}s | nfe: {self.nfe} | avg forward: {self.avg_forward_time:.4f}s"