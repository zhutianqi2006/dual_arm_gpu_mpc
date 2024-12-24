import torch
import time
import dq_torch
from dqrobotics import DQ
a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0, 0.2, 0.4, 0.5], dtype=torch.float32, device='cuda:0').repeat(1,1)
from dqrobotics import DQ
dq_robotics_a = DQ(1.0, 0.0, 0.0, 0.0, 0, 0.2, 0.4, 0.5)
print("dq_robotics_a: ", dq_robotics_a.translation())
print("torch_a: ", dq_torch.translation(a))
