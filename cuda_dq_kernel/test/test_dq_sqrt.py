import torch
import dq_mult_extension
import time
import dq_torch
from dqrobotics import DQ
a = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda:0').repeat(1,1)
from dqrobotics import DQ
dq_robotics_a = DQ(0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
print("test1——dq_robotics_a: ", dq_robotics_a.pow(0.5))
print("test1——torch_a: ", dq_torch.dq_pow(a, 0.5))
print("test1——extension_a: ", dq_mult_extension.dq_sqrt(a))
a = torch.tensor([1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=torch.float32, device='cuda:0').repeat(1,1)
dq_robotics_a = DQ(1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0)
print("test2——dq_robotics_a: ", dq_robotics_a.pow(0.5))
print("test2——torch_a: ", dq_torch.dq_pow(a, 0.5))
print("test2——extension_a: ", dq_mult_extension.dq_sqrt(a))
# test cuda dq
a = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda:0').repeat(10000,1)
dq_mult_extension.dq_sqrt(a)
dq_mult_extension.dq_sqrt(a)
start = time.time()
for i in range(1000):
    dq_mult_extension.dq_sqrt(a)
print(time.time()-start)
## test dq_torch
dq_torch.dq_pow(a, 0.5)
dq_torch.dq_pow(a, 0.5)
start = time.time()
for i in range(1000):
    dq_torch.dq_pow(a, 0.5)
print(time.time()-start)
start = time.time()
for i in range(10000000):
     dq_robotics_a.pow(0.5)
print(time.time()-start)
