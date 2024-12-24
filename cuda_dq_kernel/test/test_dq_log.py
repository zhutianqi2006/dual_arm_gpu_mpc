import torch
import time
import dq_torch
from dqrobotics import DQ
a = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.2, 1.0, 1.0, 1.0], dtype=torch.float32, device='cuda:0').repeat(1,1)
from dqrobotics import DQ
dq_robotics_a = DQ(0.5, 0.5, 0.5, 0.5, 0.2, 1.0, 1.0, 1.0)
print("dq_robotics_a: ", (dq_robotics_a.normalize()).log())
print("torch_a: ",dq_torch.dq_log(dq_torch.dq_normalize(a)))
print("dq_robotics_a: ", (dq_robotics_a.inv()))
print("torch_a: ",(dq_torch.dq_inv(a)))


