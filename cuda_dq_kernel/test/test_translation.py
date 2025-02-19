import torch
import time
import dq_torch
from dqrobotics import DQ
a = torch.tensor([9.63267947e-05,  7.07244290e-01, -7.06969239e-01, -3.67320509e-06, 3.03159877e-01,  1.23636280e-01,  1.23726146e-01, -8.79988859e-02], dtype=torch.float32, device='cuda:0').repeat(1,1)
from dqrobotics import DQ
dq_robotics_a = DQ(9.63267947e-05,  7.07244290e-01, -7.06969239e-01, -3.67320509e-06, 3.03159877e-01,  1.23636280e-01,  1.23726146e-01, -8.79988859e-02).normalize()

print("dq_robotics_a: ", dq_robotics_a.translation())
print("torch_a: ", dq_torch.dq_translation(a))
