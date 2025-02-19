#include <torch/extension.h>
#include "dq_utils.h"

torch::Tensor dq_mult(torch::Tensor q1,torch::Tensor q2){
    CHECK_INPUT(q1);
    CHECK_INPUT(q2);
    return dq_mult_cuda(q1, q2); 
}
torch::Tensor P(torch::Tensor q){
    CHECK_INPUT(q);
    return P_cuda(q);
}
torch::Tensor D(torch::Tensor q){
    CHECK_INPUT(q);
    return D_cuda(q);
}
torch::Tensor Re(torch::Tensor q){
    CHECK_INPUT(q);
    return Re_cuda(q);
}
torch::Tensor Im(torch::Tensor q){
    CHECK_INPUT(q);
    return Im_cuda(q);
}
torch::Tensor conj(torch::Tensor q){
    CHECK_INPUT(q);
    return conj_cuda(q);
}
torch::Tensor norm(torch::Tensor q){
    CHECK_INPUT(q);
    return norm_cuda(q);
}
torch::Tensor dq_translation(torch::Tensor q){
    CHECK_INPUT(q);
    return translation_cuda(q);
}
torch::Tensor rotation_angle(torch::Tensor q){
    CHECK_INPUT(q);
    return rotation_angle_cuda(q);
}
torch::Tensor rotation_axis(torch::Tensor q){
    CHECK_INPUT(q);
    return rotation_axis_cuda(q);
}
torch::Tensor dq_log(torch::Tensor q){
    CHECK_INPUT(q);
    return dq_log_cuda(q);
}
torch::Tensor dq_exp(torch::Tensor q){
    CHECK_INPUT(q);
    return dq_exp_cuda(q);
}
torch::Tensor dq_sqrt(torch::Tensor q){
    CHECK_INPUT(q);
    return dq_sqrt_cuda(q);
}
torch::Tensor dq_inv(torch::Tensor q){
    CHECK_INPUT(q);
    return dq_inv_cuda(q);
}
torch::Tensor dq_normalize(torch::Tensor q){
    CHECK_INPUT(q);
    return dq_normalize_cuda(q);
}
torch::Tensor Ad(torch::Tensor q1, torch::Tensor q2)
{
    CHECK_INPUT(q1);
    CHECK_INPUT(q2);
    return Ad_cuda(q1,q2);
}
torch::Tensor hamiplus8(torch::Tensor q)
{
    CHECK_INPUT(q);
    return hamiplus8_cuda(q);
}
torch::Tensor haminus8(torch::Tensor q)
{
    CHECK_INPUT(q);
    return haminus8_cuda(q);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_abs_pose_rel_jac(torch::Tensor dh1, torch::Tensor dh2,
torch::Tensor base1, torch::Tensor base2,
torch::Tensor effector1, torch::Tensor effector2,
torch::Tensor theta1, torch::Tensor theta2,
torch::Tensor line_d, torch::Tensor quat_line_ref,
int ith1, int ith2 ,int dh1_type, int dh2_type)
{
    CHECK_INPUT(dh1);
    CHECK_INPUT(dh2);
    CHECK_INPUT(base1);
    CHECK_INPUT(base2);
    CHECK_INPUT(effector1);
    CHECK_INPUT(effector2);
    CHECK_INPUT(theta1);
    CHECK_INPUT(theta2);
    return rel_abs_pose_rel_jac_cuda(dh1, dh2, base1, base2, effector1, effector2, theta1, theta2, line_d, quat_line_ref, ith1, ith2, dh1_type ,dh2_type);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_abs_pose_rel_abs_jac(torch::Tensor dh1, torch::Tensor dh2,
torch::Tensor base1, torch::Tensor base2,
torch::Tensor effector1, torch::Tensor effector2,
torch::Tensor theta1, torch::Tensor theta2,
torch::Tensor line_d, torch::Tensor quat_line_ref,
int ith1, int ith2 ,int dh1_type, int dh2_type)
{
    CHECK_INPUT(dh1);
    CHECK_INPUT(dh2);
    CHECK_INPUT(base1);
    CHECK_INPUT(base2);
    CHECK_INPUT(effector1);
    CHECK_INPUT(effector2);
    CHECK_INPUT(theta1);
    CHECK_INPUT(theta2);
    return rel_abs_pose_rel_abs_jac_cuda(dh1, dh2, base1, base2, effector1, effector2, theta1, theta2, line_d, quat_line_ref, ith1, ith2, dh1_type ,dh2_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dq_mult", &dq_mult);
    m.def("P", &P);
    m.def("D", &D);
    m.def("Re", &Re);
    m.def("Im", &Im);
    m.def("conj", &conj);
    m.def("norm", &norm);
    m.def("dq_translation", &dq_translation);
    m.def("rotation_angle", &rotation_angle);
    m.def("rotation_axis", &rotation_axis);
    m.def("dq_log", &dq_log);
    m.def("dq_exp", &dq_exp);
    m.def("dq_sqrt", &dq_sqrt);
    m.def("dq_inv", &dq_inv);
    m.def("dq_normalize", &dq_normalize);
    m.def("Ad", &Ad);
    m.def("hamiplus8", &hamiplus8);
    m.def("haminus8", &haminus8);
    m.def("rel_abs_pose_rel_jac", &rel_abs_pose_rel_jac);
    m.def("rel_abs_pose_rel_abs_jac", &rel_abs_pose_rel_abs_jac);
}
