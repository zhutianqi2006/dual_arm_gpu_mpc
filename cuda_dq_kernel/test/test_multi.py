import torch
import dq_mult_extension
import time
import dq_torch
a = torch.tensor([0, 2, 3, 4, 0, 6, 7, 8], dtype=torch.float32, device='cuda:0').repeat(1000,1)
b = torch.tensor([0, 2, 3, 4, 0, 6, 7, 8], dtype=torch.float32, device='cuda:0').repeat(1000,1)
dq_mult_extension.dq_mult(a, b)
dq_mult_extension.dq_mult(a, b)
start = time.time()
for i in range(1000):
    dq_mult_extension.dq_mult(a, b)
print(time.time()-start)
print(dq_mult_extension.dq_mult(a, b))

dq_torch.dq_mult(a, b)
dq_torch.dq_mult(a, b)
print(dq_torch.dq_mult(a, b))
start = time.time()
for i in range(1000):
    dq_torch.dq_mult(a, b)
print(time.time()-start)

from dqrobotics import DQ
a = DQ(0, 2, 3, 4, 0, 6, 7, 8)
b = DQ(0, 2, 3, 4, 0, 6, 7, 8)
start = time.time()
for i in range(1000):
    c = a*b
print(time.time()-start)
