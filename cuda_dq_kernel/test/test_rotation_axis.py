import torch
import dq_mult_extension
import time
import dq_torch
from dqrobotics import DQ
a = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0, 0, 0], dtype=torch.float32, device='cuda:0').repeat(1,1)
from dqrobotics import DQ
dq_robotics_a = DQ(0.5, 0.5, 0.5, 0.5, 0.0, 0, 0,0)
print("dq_robotics_a: ", dq_robotics_a.rotation_axis())
print("torch_a: ", dq_torch.rotation_axis(a))
print("extension_a: ", dq_mult_extension.rotation_axis(a))

# test cuda dq
a = torch.tensor([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0], dtype=torch.float32, device='cuda:0').repeat(1000,1)
dq_mult_extension.rotation_axis(a)
dq_mult_extension.rotation_axis(a)
start = time.time()
for i in range(1000):
    dq_mult_extension.rotation_axis(a)
print(time.time()-start)
## test dq_torch
dq_torch.rotation_axis(a)
dq_torch.rotation_axis(a)
start = time.time()
for i in range(1000):
    dq_torch.rotation_axis(a)
print(time.time()-start)
