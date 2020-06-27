import torch 
from contextlib import contextmanager
import gc

class DeviceManager:
    def __init__(self):
        self.rounding_digits = 3
        
    def available_gpus_info(self):
        gpus_count = torch.cuda.device_count()
        print('Number of GPUs : ', gpus_count, '\n')
        
        for i in range(gpus_count):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_name = gpu_props.name
            gpu_memory = round(gpu_props.total_memory * 1e-9, self.rounding_digits)
            
            print('* GPU index : {} \t GPU name : {} \t RAM : {} [GB] \n'.format(i, gpu_name, gpu_memory))
    
    def get_gpu_device(self, gpu_id):
        device =  torch.device("cuda:{}".format(gpu_id))
        return device
    
    def get_gpu_memory_allocated(self):
        gpu_memory_allocated = round(torch.cuda.memory_allocated() * 1e-6, self.rounding_digits)
        print('Current GPU memory allocated {} [MB] GPU RAM'.format(gpu_memory_allocated))
        
    @contextmanager
    def get_last_gpu_usage(self, desc_str=' '):
        pre_gpu_memory_allocated = torch.cuda.memory_allocated()
        yield None
        post_gpu_memory_allocated = torch.cuda.memory_allocated()
        last_gpu_memory_allocated = (post_gpu_memory_allocated - pre_gpu_memory_allocated) * 1e-6
        last_gpu_memory_allocated = round(last_gpu_memory_allocated, self.rounding_digits)
        print(desc_str + ' reserved {} [MB] GPU RAM'.format(last_gpu_memory_allocated))
        
    def tensors_tracking(self):
      for obj in gc.get_objects():
          try:
              if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                  print(type(obj), obj.size())
          except: 
              pass

