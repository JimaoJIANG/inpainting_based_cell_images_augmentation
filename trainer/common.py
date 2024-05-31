import torch

class timer():
    def __init__(self):
        self.acc = 0
        self.t0 = torch.cuda.Event(enable_timing=True)
        self.t1 = torch.cuda.Event(enable_timing=True)
        self.tic()

    def tic(self):
        self.t0.record()

    def toc(self, restart=False):
        self.t1.record()
        torch.cuda.synchronize()
        diff = self.t0.elapsed_time(self.t1) /1000.
        if restart: self.tic()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0
            