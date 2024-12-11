import torch
import dq_mult_extension
import time
import dq_torch
from dqrobotics import DQ
a = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.3, 0.2], dtype=torch.float32, device='cuda:0').repeat(1,1)
b = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda:0').repeat(1,1)
from dqrobotics import DQ
dq_robotics_a = DQ(0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.3, 0.2)
dq_robotics_b = DQ(0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
print("dq_robotics_a: ", dq_robotics_a.Ad(dq_robotics_b))
print("torch_a: ", dq_torch.Ad(a,b))
print("extension_a: ", dq_mult_extension.Ad(a,b))

# test cuda dq
a = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.2], dtype=torch.float32, device='cuda:0').repeat(1000,1)
b = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda:0').repeat(1000,1)
dq_mult_extension.Ad(a,b)
dq_mult_extension.Ad(a,b)
start = time.time()
for i in range(1000):
    dq_mult_extension.Ad(a,b)
print(time.time()-start)
## test dq_torch
dq_torch.Ad(a,b)
dq_torch.Ad(a,b)
start = time.time()
for i in range(1000):
    dq_torch.Ad(a,b)
print(time.time()-start)
