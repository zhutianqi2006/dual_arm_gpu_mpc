#include<torch/extension.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dq_mult_cuda(torch::Tensor q1,torch::Tensor q2);
torch::Tensor P_cuda(torch::Tensor q);
torch::Tensor D_cuda(torch::Tensor q);
torch::Tensor Re_cuda(torch::Tensor q);
torch::Tensor Im_cuda(torch::Tensor q);
torch::Tensor conj_cuda(torch::Tensor q);
torch::Tensor norm_cuda(torch::Tensor q);
torch::Tensor translation_cuda(torch::Tensor q);
torch::Tensor rotation_angle_cuda(torch::Tensor q);
torch::Tensor rotation_axis_cuda(torch::Tensor q);
torch::Tensor dq_log_cuda(torch::Tensor q);
torch::Tensor dq_exp_cuda(torch::Tensor q);
torch::Tensor dq_sqrt_cuda(torch::Tensor q);
torch::Tensor dq_inv_cuda(torch::Tensor q);
torch::Tensor dq_normalize_cuda(torch::Tensor q);
torch::Tensor Ad_cuda(torch::Tensor q1,torch::Tensor q2);
torch::Tensor hamiplus8_cuda(torch::Tensor q);
torch::Tensor haminus8_cuda(torch::Tensor q);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_abs_pose_rel_jac_cuda
(torch::Tensor dh1, torch::Tensor dh2,
torch::Tensor base1, torch::Tensor base2,
torch::Tensor effector1, torch::Tensor effector2,
torch::Tensor theta1, torch::Tensor theta2,
torch::Tensor line_d, torch::Tensor quat_line_ref,
int ith1, int ith2, int dh1_type, int dh2_type);