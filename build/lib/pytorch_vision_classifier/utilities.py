from contextlib import contextmanager
import torch
import time

class ElapsedTime:
    def __init__(self, message, times_list = None, fp_digits=6, show_time = True):
        self.times_list = times_list
        self.fp_digits = fp_digits
        self.show_time = show_time
        self.message = message
    
    def __show_time(self, e):
        
        if self.show_time:
            if e >= 1:
                print('{} takes {:.0f} [m] {:.0f} [s]'.format(self.message, e // 60, e % 60))
            else:
                print('{} takes {} [s]'.format(self.message, e))
    
    def __save_time(self, e):
        if type(self.times_list) is list:
            self.times_list.append(e)
    
    @contextmanager
    def cpu(self, with_gpu=False):
        s = time.perf_counter()
        yield None
        
        if with_gpu:
            torch.cuda.synchronize()
        
        e = round(time.perf_counter() - s, self.fp_digits)

        self.__save_time(e)
        self.__show_time(e)
    
    @contextmanager
    def gpu(self):
        # Define events
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        yield None

        torch.cuda.synchronize()
        e.record()
        torch.cuda.synchronize()
        
        e = round(s.elapsed_time(e)/1000, self.fp_digits)
        
        self.__save_time(e)
        self.__show_time(e)
        
    @staticmethod
    def consume_gpu(n, device):
        """ Dummy function for test purpose """
        a = torch.ones((n,n), device=device)
        b = torch.ones((n,n), device=device)
        c = a * b
