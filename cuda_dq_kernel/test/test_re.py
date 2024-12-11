import torch
import dq_mult_extension
import time
import dq_torch
from dqrobotics import DQ
a = torch.tensor([1, 2, 3, 4, 0, 6, 7, 8], dtype=torch.float32, device='cuda:0').repeat(1,1)
from dqrobotics import DQ
dq_robotics_a = DQ(1, 2, 3, 4, 0, 6, 7, 8)
print("dq_robotics_a: ", dq_robotics_a.Re())
print("torch_a: ", dq_torch.Re(a))
print("extension_a: ", dq_mult_extension.Re(a))

# test cuda dq
a = torch.tensor([1, 2, 3, 4, 0, 6, 7, 8], dtype=torch.float32, device='cuda:0').repeat(1000,1)
dq_mult_extension.Re(a)
dq_mult_extension.Re(a)
start = time.time()
for i in range(1000):
    dq_mult_extension.Re(a)
print(time.time()-start)
## test dq_torch
dq_torch.Re(a)
dq_torch.Re(a)
start = time.time()
for i in range(1000):
    dq_torch.Re(a)
print(time.time()-start)
