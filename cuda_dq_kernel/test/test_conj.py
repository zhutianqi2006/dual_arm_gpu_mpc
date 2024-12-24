import torch
import time
import dq_torch
from dqrobotics import DQ
a = torch.tensor([0, 2, 3, 4, 1, 6, 7, 8], dtype=torch.float32, device='cuda:0').repeat(1,1)
from dqrobotics import DQ
dq_robotics_a = DQ(0, 2, 3, 4, 1, 6, 7, 8)
print(dq_robotics_a)
print("dq_robotics_a: ", dq_robotics_a.conj())
print("torch_a: ", dq_torch.conj(a))

