#include <torch/extension.h>
#include <stdio.h>
//----------------------------- inline 函数------------------------------------
//----------------------------- inline 函数------------------------------------
//----------------------------- inline 函数------------------------------------
template <typename scalar_t>
__device__ __forceinline__ void mat_mul_inline(const scalar_t* A, const scalar_t* B, scalar_t* C, int M, int K, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            scalar_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void mat_mul88_inline(const scalar_t* A, const scalar_t* B, scalar_t* C) {
    const int M = 8;
    const int K = 8;
    const int N = 8;

    // 遍历结果矩阵的每个元素
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            scalar_t sum = 0;
            // 计算行列乘积的累加和
            for (int k = 0; k < K; ++k) {
                // 注意这里的索引方式进行了修改，以适应一维数组
                sum += A[row * K + k] * B[k * N + col];
            }
            // 存储结果，同样注意索引方式
            C[row * N + col] = sum;
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void mat_mul86_inline(const scalar_t* A, const scalar_t* B, scalar_t* C) {
    const int M = 8;
    const int K = 8;
    const int N = 6;

    // 遍历结果矩阵的每个元素
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            scalar_t sum = 0;
            // 计算行列乘积的累加和
            for (int k = 0; k < K; ++k) {
                // 注意这里的索引方式进行了修改，以适应一维数组
                sum += A[row * K + k] * B[k * N + col];
            }
            // 存储结果，同样注意索引方式
            C[row * N + col] = sum;
        }
    }
}

template <typename scalar_t>
__device__ __forceinline__ void dq_mult_inline(
    const scalar_t* q1, const scalar_t* q2, scalar_t* result)
{
    // 实部四元数的提取
    scalar_t a1 = q1[0], b1 = q1[1], c1 = q1[2], d1 = q1[3];
    scalar_t ad1 = q1[4], bd1 = q1[5], cd1 = q1[6], dd1 = q1[7];

    // 双部四元数的提取
    scalar_t a2 = q2[0], b2 = q2[1], c2 = q2[2], d2 = q2[3];
    scalar_t ad2 = q2[4], bd2 = q2[5], cd2 = q2[6], dd2 = q2[7];

    // 实部四元数乘法
    result[0] = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2;
    result[1] = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2;
    result[2] = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2;
    result[3] = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2;

    // 双部四元数乘法
    result[4] = ad1 * a2 + a1 * ad2 - bd1 * b2 - b1 * bd2 - cd1 * c2 - c1 * cd2 - dd1 * d2 - d1 * dd2;
    result[5] = ad1 * b2 + a1 * bd2 + bd1 * a2 + b1 * ad2 + cd1 * d2 + c1 * dd2 - dd1 * c2 - d1 * cd2;
    result[6] = ad1 * c2 + a1 * cd2 - bd1 * d2 - b1 * dd2 + cd1 * a2 + c1 * ad2 + dd1 * b2 + d1 * bd2;
    result[7] = ad1 * d2 + a1 * dd2 + bd1 * c2 + b1 * cd2 - cd1 * b2 - c1 * bd2 + dd1 * a2 + d1 * ad2;
}

template <typename scalar_t>
__device__ __forceinline__ void P_inline(const scalar_t* v, scalar_t* translation) {
    // 示例：假设平移部分存储在 v 的后四位
    for (int i = 0; i < 4; i++) {
        translation[i] = v[i];
    }
}

// 假设 D(v) 返回四元数的旋转部分
template <typename scalar_t>
__device__ __forceinline__ void D_inline(const scalar_t* v, scalar_t* rotation) {
    // 示例：假设旋转部分存储在 v 的前四位
    for (int i = 0; i < 4; i++) {
        rotation[i] = v[4+i];
    }
}

// 计算四元数的共轭
template <typename scalar_t>
__device__ __forceinline__ void conj_inline(const scalar_t* q, scalar_t* result) {
    // 计算四元数的共轭，只反转虚部
    result[0] = q[0];
    result[1] = -q[1];
    result[2] = -q[2];
    result[3] = -q[3];
    // 双四元数的第二部分也需要处理
    result[4] = q[4];
    result[5] = -q[5];
    result[6] = -q[6];
    result[7] = -q[7];
}

// 计算四元数的共轭
template <typename scalar_t>
__device__ __forceinline__ void qconj_inline(const scalar_t* q, scalar_t* result) {
    // 计算四元数的共轭，只反转虚部
    result[0] = q[0];
    result[1] = -q[1];
    result[2] = -q[2];
    result[3] = -q[3];
}

// 计算四元数的共轭
template <typename scalar_t>
__device__ __forceinline__ void dq_position_inline(const scalar_t* q, scalar_t* result) {
    scalar_t a1 = q[4], b1 = q[5], c1 = q[6], d1 = q[7];
    scalar_t a2 = q[0], b2 = -q[1], c2 = -q[2], d2 = -q[3];
    // 返回位置
    result[0] = 2.0*(a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2);
    result[1] = 2.0*(a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2);
    result[2] = 2.0*(a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2);
}

template <typename scalar_t>
__device__ __forceinline__ void q_mult_inline(
    const scalar_t* q1, const scalar_t* q2, scalar_t* result)
{
    scalar_t a1 = q1[0], b1 = q1[1], c1 = q1[2], d1 = q1[3];
    scalar_t a2 = q2[0], b2 = q2[1], c2 = q2[2], d2 = q2[3];
    // 实部四元数乘法
    result[0] = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2;
    result[1] = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2;
    result[2] = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2;
    result[3] = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2;
}

template <typename scalar_t>
__device__ __forceinline__ void q_dot_inline(
    const scalar_t* q1, const scalar_t* q2, scalar_t* result)
{
    scalar_t a1 = q1[0], b1 = q1[1], c1 = q1[2], d1 = q1[3];
    scalar_t a2 = q2[0], b2 = q2[1], c2 = q2[2], d2 = q2[3];
    // 实部四元数乘法
    result[0] = a1 * a2 + b1 * b2 + c1 * c2 + d1 * d2;
}

template <typename scalar_t>
__device__ __forceinline__ void q_angle_inline(
    const scalar_t* q1, const scalar_t* q2, scalar_t* result)
{
    scalar_t a1 = q1[0], b1 = q1[1], c1 = q1[2], d1 = q1[3];
    scalar_t a2 = q2[0], b2 = q2[1], c2 = q2[2], d2 = q2[3];
    // 实部四元数乘法
    result[0] = 57.2958*acos(a1 * a2 + b1 * b2 + c1 * c2 + d1 * d2);
}

template <typename scalar_t>
__device__ __forceinline__ void norm_inline(const scalar_t* q, scalar_t* result) {
    scalar_t aux[8] = {0.0};
    for (int i = 0; i < 8; i++) {
        aux[i] = q[i];  // 复制输入数组
    }
    scalar_t primary[8] = {0.0};
    P_inline(aux, primary);
    bool primary_is_zero = true;
    for (int i = 0; i < 4; i++) {
        if (primary[i] != 0) {
            primary_is_zero = false;
            break;
        }
    }
    if (primary_is_zero) {
        for (int i = 0; i < 8; i++) {
            result[i] = 0; 
        }
    } else {
        scalar_t conj_aux[8];
        conj_inline(aux, conj_aux); 
        dq_mult_inline(conj_aux, aux, result);  // 双四元数乘法

        result[0] = sqrt(result[0]);  // 计算实部的平方根
        result[4] = result[4] / (2 * result[0]);  // 计算双部除以两倍实部的平方根
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t rotation_angle_inline(const scalar_t* q) {
    // 根据四元数计算旋转角度，这里假设 q[0] 是实部
    scalar_t clamped_q0 = max(min(q[0], scalar_t(1)), scalar_t(-1));
    return 2.0 * acos(clamped_q0);
}

template <typename scalar_t>
__device__ __forceinline__ void Im_inline(const scalar_t* q, scalar_t* im_q) {
    // 初始化为零
    im_q[0] = 0;
    im_q[4] = 0;
    // 复制虚部
    for (int i = 1; i < 4; i++) { // 第二到第四元素
        im_q[i] = q[i];
    }
    for (int i = 5; i < 8; i++) { // 第六到第八元素
        im_q[i] = q[i];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void rotation_axis_inline(const scalar_t* v, scalar_t* rot_axis) {
    scalar_t phi_tensor[8] = {0.0};
    scalar_t p_v[8] = {0.0};
    scalar_t im_rot_axis[8] = {0.0};
    // 计算旋转角度的一半
    scalar_t phi = rotation_angle_inline(v) / 2.0;
    scalar_t sin_phi = sin(phi);
    if (fabs(sin_phi) < 1e-8) {
        sin_phi = 1e-8;
    }
    phi_tensor[0] = 1.0 / sin_phi;
    for (int i = 1; i < 8; i++) {
        phi_tensor[i] = 0; // 或者根据需要设置适当的值
    }
    P_inline(v, p_v);
    Im_inline(p_v, im_rot_axis);
    dq_mult_inline(phi_tensor, im_rot_axis, rot_axis);
}

template <typename scalar_t>
__device__ __forceinline__ void translation_inline(const scalar_t* v, scalar_t* translation_result) {
    // 获取 P(v)
    scalar_t p_v[8] ={0.0};
    P_inline(v, p_v);

    // 获取 D(v)
    scalar_t d_v[8] ={0.0};
    D_inline(v, d_v);

    // 获取 conj(P(v))
    scalar_t conj_p_v[8] ={0.0};
    conj_inline(p_v, conj_p_v);

    // 计算 translation = 2.0 * dq_mult(D(v), conj(P(v)))
    scalar_t temp_result[8] ={0.0};
    dq_mult_inline(d_v, conj_p_v, temp_result);

    for (int i = 0; i < 8; i++) {
        translation_result[i] = 2.0 * temp_result[i];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void dq_log_inline(const scalar_t* v, scalar_t* result) {
    scalar_t phi = rotation_angle_inline(v);
    scalar_t phi_tensor[8] = {0.5 * phi, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    scalar_t axis[8] ={0.0};
    rotation_axis_inline(v, axis);
    scalar_t p[8] ={0.0};
    dq_mult_inline(phi_tensor, axis, p);
    scalar_t d[8]={0.0};
    translation_inline(v, d);
    for (int i = 0; i < 4; i++) {
        d[i] *= 0.5;
    }
    for (int i = 0; i < 8; i++) {
        result[i] = (i < 4) ? p[i] : d[i - 4];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void dq_exp_inline(const scalar_t* v, scalar_t* result) {
    scalar_t prim[8] = {0.0};
    P_inline(v, prim);
    scalar_t phi = 0.0;
    for (int i = 0; i < 8; i++) {
        phi += prim[i] * prim[i];
    }
    phi = sqrt(phi);

    scalar_t phi_tensor[8] = {sin(phi)/(phi + 1e-8), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    scalar_t new_prim[8] = {0.0};
    P_inline(v, new_prim);
    dq_mult_inline(phi_tensor, new_prim, new_prim);
    new_prim[0] += cos(phi);

    scalar_t E_[8] = {0, 0, 0, 0, 1, 0, 0, 0};
    scalar_t d[8]= {0.0};
    D_inline(v, d);
    scalar_t temp[8]= {0.0};
    scalar_t temp2[8]= {0.0};
    dq_mult_inline(E_, d, temp);
    dq_mult_inline(temp, new_prim, temp2);
    for (int i = 0; i < 8; i++) {
        result[i] = temp2[i] + new_prim[i];
    }
}

template <typename scalar_t>
__device__ __forceinline__ void dq_sqrt_inline(const scalar_t* v, scalar_t* result){
    scalar_t dq_log_value[8] = {0.0};
    dq_log_inline(v, dq_log_value);
    for (int i = 0; i < 8; i++) {
            dq_log_value[i] *= 0.5;
    }
    dq_exp_inline(dq_log_value, result);
}


template <typename scalar_t>
__device__ __forceinline__ void dq_inv_inline(const scalar_t* v, scalar_t* result) {
    scalar_t v_conj[8] = {0.0};
    scalar_t aux[8] = {0.0};
    scalar_t inv_dq_temp[8] = {0.0};
    conj_inline(v, v_conj);
    dq_mult_inline(v, v_conj, aux);
    inv_dq_temp[0] = 1.0 / aux[0];
    inv_dq_temp[4] = -aux[4] / pow(aux[0], 2);
    dq_mult_inline(v_conj, inv_dq_temp, result);
}

template <typename scalar_t>
__device__ __forceinline__ void dq_normalize_inline(const scalar_t* v, scalar_t* result) {
    scalar_t dq_norm_value[8] = {0.0};
    norm_inline(v, dq_norm_value);
    scalar_t dq_inv_value[8] = {0.0};
    dq_inv_inline(dq_norm_value, dq_inv_value);
    dq_mult_inline(v, dq_inv_value, result);
}

template <typename scalar_t>
__device__ __forceinline__ void hamiplus4_inline(const scalar_t* v, scalar_t* result) 
{
    result[0]  =  v[0]; result[1]  = -v[1]; result[2]  = -v[2]; result[3]  = -v[3];
    result[4]  =  v[1]; result[5]  =  v[0]; result[6]  = -v[3]; result[7]  =  v[2];
    result[8]  =  v[2]; result[9]  =  v[3]; result[10] =  v[0]; result[11] = -v[1];
    result[12] =  v[3]; result[13] = -v[2]; result[14] =  v[1]; result[15] =  v[0];

}

template <typename scalar_t>
__device__ __forceinline__ void haminus4_inline(const scalar_t* v, scalar_t* result)
{ 
    // result[0][0] = v[0]; result[0][1] = -v[1]; result[0][2] = -v[2]; result[0][3] = -v[3];
    // result[1][0] = v[1]; result[1][1] =  v[0]; result[1][2] =  v[3]; result[1][3] = -v[2];
    // result[2][0] = v[2]; result[2][1] = -v[3]; result[2][2] =  v[0]; result[2][3] =  v[1];
    // result[3][0] = v[3]; result[3][1] =  v[2]; result[3][2] = -v[1]; result[3][3] =  v[0];
    result[0]  =  v[0]; result[1]  = -v[1]; result[2]  = -v[2]; result[3]  = -v[3];
    result[4]  =  v[1]; result[5]  =  v[0]; result[6]  =  v[3]; result[7]  = -v[2];
    result[8]  =  v[2]; result[9]  = -v[3]; result[10] =  v[0]; result[11] =  v[1];
    result[12] =  v[3]; result[13] =  v[2]; result[14] = -v[1]; result[15] =  v[0];
}

template <typename scalar_t>
__device__ __forceinline__ void hamiplus8_inline_v0(const scalar_t* v, scalar_t** result) 
{
    // Zero-initialize the matrix
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            result[i][j] = 0;

    // First 4x4 block
    result[0][0] =  v[0]; result[0][1] = -v[1]; result[0][2] = -v[2]; result[0][3] = -v[3];
    result[1][0] =  v[1]; result[1][1] =  v[0]; result[1][2] = -v[3]; result[1][3] =  v[2];
    result[2][0] =  v[2]; result[2][1] =  v[3]; result[2][2] =  v[0]; result[2][3] = -v[1];
    result[3][0] =  v[3]; result[3][1] = -v[2]; result[3][2] =  v[1]; result[3][3] =  v[0];

    // Third 4x4 block (bottom left)
    result[4][0] =  v[4]; result[4][1] = -v[5]; result[4][2] = -v[6]; result[4][3] = -v[7];
    result[5][0] =  v[5]; result[5][1] =  v[4]; result[5][2] = -v[7]; result[5][3] =  v[6];
    result[6][0] =  v[6]; result[6][1] =  v[7]; result[6][2] =  v[4]; result[6][3] = -v[5];
    result[7][0] =  v[7]; result[7][1] = -v[6]; result[7][2] =  v[5]; result[7][3] =  v[4];

    // Fourth 4x4 block (bottom right)
    result[4][4] =  v[0]; result[4][5] = -v[1]; result[4][6] = -v[2]; result[4][7] = -v[3];
    result[5][4] =  v[1]; result[5][5] =  v[0]; result[5][6] = -v[3]; result[5][7] =  v[2];
    result[6][4] =  v[2]; result[6][5] =  v[3]; result[6][6] =  v[0]; result[6][7] = -v[1];
    result[7][4] =  v[3]; result[7][5] = -v[2]; result[7][6] =  v[1]; result[7][7] =  v[0];
}

template <typename scalar_t>
__device__ __forceinline__ void hamiplus8_inline(const scalar_t* v, scalar_t* result) 
{   
    // 0-7 one line
    result[0]  =  v[0]; result[1]  = -v[1]; result[2]  = -v[2]; result[3]  = -v[3];
    result[4]  =     0; result[5]  =     0; result[6]  =     0; result[7]  =     0;
    result[8]  =  v[1]; result[9]  =  v[0]; result[10] = -v[3]; result[11] =  v[2];
    result[12] =     0; result[13] =     0; result[14] =     0; result[15] =     0;
    result[16] =  v[2]; result[17] =  v[3]; result[18] =  v[0]; result[19] = -v[1];
    result[20] =     0; result[21] =     0; result[22] =     0; result[23] =     0;
    result[24] =  v[3]; result[25] = -v[2]; result[26] =  v[1]; result[27] =  v[0];
    result[28] =     0; result[29] =     0; result[30] =     0; result[31] =     0;
    result[32] =  v[4]; result[33] = -v[5]; result[34] = -v[6]; result[35] = -v[7];
    result[36] =  v[0]; result[37] = -v[1]; result[38] = -v[2]; result[39] = -v[3];
    result[40] =  v[5]; result[41] =  v[4]; result[42] = -v[7]; result[43] =  v[6];
    result[44] =  v[1]; result[45] =  v[0]; result[46] = -v[3]; result[47] =  v[2];
    result[48] =  v[6]; result[49] =  v[7]; result[50] =  v[4]; result[51] = -v[5];
    result[52] =  v[2]; result[53] =  v[3]; result[54] =  v[0]; result[55] = -v[1];
    result[56] =  v[7]; result[57] = -v[6]; result[58] =  v[5]; result[59] =  v[4];
    result[60] =  v[3]; result[61] = -v[2]; result[62] =  v[1]; result[63] =  v[0];
}

template <typename scalar_t>
__device__ __forceinline__ void haminus8_inline_v0(const scalar_t* v, scalar_t** result) 
{    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            result[i][j] = 0;

    // First 4x4 block
    result[0][0] =  v[0]; result[0][1] = -v[1]; result[0][2] = -v[2]; result[0][3] = -v[3];
    result[1][0] =  v[1]; result[1][1] =  v[0]; result[1][2] =  v[3]; result[1][3] = -v[2];
    result[2][0] =  v[2]; result[2][1] = -v[3]; result[2][2] =  v[0]; result[2][3] =  v[1];
    result[3][0] =  v[3]; result[3][1] =  v[2]; result[3][2] = -v[1]; result[3][3] =  v[0];

    // Third 4x4 block (bottom left)
    result[4][0] =  v[4]; result[4][1] = -v[5]; result[4][2] = -v[6]; result[4][3] = -v[7];
    result[5][0] =  v[5]; result[5][1] =  v[4]; result[5][2] =  v[7]; result[5][3] = -v[6];
    result[6][0] =  v[6]; result[6][1] = -v[7]; result[6][2] =  v[4]; result[6][3] =  v[5];
    result[7][0] =  v[7]; result[7][1] =  v[6]; result[7][2] = -v[5]; result[7][3] =  v[4];

    // Fourth 4x4 block (bottom right)
    result[4][4] =  v[0]; result[4][5] = -v[1]; result[4][6] = -v[2]; result[4][7] = -v[3];
    result[5][4] =  v[1]; result[5][5] =  v[0]; result[5][6] =  v[3]; result[5][7] = -v[2];
    result[6][4] =  v[2]; result[6][5] = -v[3]; result[6][6] =  v[0]; result[6][7] =  v[1];
    result[7][4] =  v[3]; result[7][5] =  v[2]; result[7][6] = -v[1]; result[7][7] =  v[0];
}

template <typename scalar_t>
__device__ __forceinline__ void haminus8_inline(const scalar_t* v, scalar_t* result)
{
    // 0-7 one line
    result[0]  =  v[0]; result[1]  = -v[1]; result[2]  = -v[2]; result[3]  = -v[3];
    result[4]  =     0; result[5]  =     0; result[6]  =     0; result[7]  =     0;
    result[8]  =  v[1]; result[9]  =  v[0]; result[10] =  v[3]; result[11] = -v[2];
    result[12] =     0; result[13] =     0; result[14] =     0; result[15] =     0;
    result[16] =  v[2]; result[17] = -v[3]; result[18] =  v[0]; result[19] =  v[1];
    result[20] =     0; result[21] =     0; result[22] =     0; result[23] =     0;
    result[24] =  v[3]; result[25] =  v[2]; result[26] = -v[1]; result[27] =  v[0];
    result[28] =     0; result[29] =     0; result[30] =     0; result[31] =     0;
    result[32] =  v[4]; result[33] = -v[5]; result[34] = -v[6]; result[35] = -v[7];
    result[36] =  v[0]; result[37] = -v[1]; result[38] = -v[2]; result[39] = -v[3];
    result[40] =  v[5]; result[41] =  v[4]; result[42] =  v[7]; result[43] = -v[6];
    result[44] =  v[1]; result[45] =  v[0]; result[46] =  v[3]; result[47] = -v[2];
    result[48] =  v[6]; result[49] = -v[7]; result[50] =  v[4]; result[51] =  v[5];
    result[52] =  v[2]; result[53] = -v[3]; result[54] =  v[0]; result[55] =  v[1];
    result[56] =  v[7]; result[57] =  v[6]; result[58] = -v[5]; result[59] =  v[4];
    result[60] =  v[3]; result[61] =  v[2]; result[62] = -v[1]; result[63] =  v[0];
} 

template <typename scalar_t>
__device__ __forceinline__ void Ad_inline(const scalar_t* q1, const scalar_t* q2, scalar_t* result) 
{
    scalar_t conj_q1[8] = {0.0};
    scalar_t q1q2[8]= {0.0};
    conj_inline(q1, conj_q1);
    dq_mult_inline(q1, q2, q1q2);  // Multiply q1 and q2
    dq_mult_inline(q1q2, conj_q1, result);  // Multiply result by conjugate of q1
}

template <typename scalar_t>
__device__ __forceinline__ void dh2dq_inline(const scalar_t* dh, const scalar_t* theta,
                                             int ith, scalar_t* q)
{
    scalar_t half_theta = dh[0 + ith * 5] / 2.0 + theta[ith] / 2.0;
    scalar_t d = dh[1 + ith * 5];
    scalar_t a = dh[2 + ith * 5];
    scalar_t half_alpha = dh[3 + ith * 5] / 2.0;

    scalar_t sin_half_theta = sinf(half_theta);
    scalar_t cos_half_theta = cosf(half_theta);
    scalar_t sin_half_alpha = sinf(half_alpha);
    scalar_t cos_half_alpha = cosf(half_alpha);

    scalar_t a_cos_half_alpha = a * cos_half_alpha;
    scalar_t d_sin_half_alpha = d * sin_half_alpha;
    scalar_t a_sin_half_alpha = a * sin_half_alpha;
    scalar_t d_cos_half_alpha = d * cos_half_alpha;

    q[0] = cos_half_alpha * cos_half_theta;
    q[1] = sin_half_alpha * cos_half_theta;
    q[2] = sin_half_alpha * sin_half_theta;
    q[3] = cos_half_alpha * sin_half_theta;
    q[4] = -(a_sin_half_alpha * cos_half_theta + d_cos_half_alpha * sin_half_theta) / 2.0;
    q[5] =  (a_cos_half_alpha * cos_half_theta - d_sin_half_alpha * sin_half_theta) / 2.0;
    q[6] =  (a_cos_half_alpha * sin_half_theta + d_sin_half_alpha * cos_half_theta) / 2.0;
    q[7] =  (d_cos_half_alpha * cos_half_theta - a_sin_half_alpha * sin_half_theta) / 2.0;
}

template <typename scalar_t>
__device__ __forceinline__ void mdh2dq_inline(const scalar_t* dh, const scalar_t* theta,
                                             int ith, scalar_t* q)
{
    scalar_t half_theta = dh[0 + ith * 5] / 2.0 + theta[ith] / 2.0;
    scalar_t d = dh[1 + ith * 5];
    scalar_t a = dh[2 + ith * 5];
    scalar_t half_alpha = dh[3 + ith * 5] / 2.0;

    scalar_t sin_half_theta = sinf(half_theta);
    scalar_t cos_half_theta = cosf(half_theta);
    scalar_t sin_half_alpha = sinf(half_alpha);
    scalar_t cos_half_alpha = cosf(half_alpha);

    scalar_t a_cos_half_alpha = a * cos_half_alpha;
    scalar_t d_sin_half_alpha = d * sin_half_alpha;
    scalar_t a_sin_half_alpha = a * sin_half_alpha;
    scalar_t d_cos_half_alpha = d * cos_half_alpha;

    q[0] = cos_half_alpha * cos_half_theta;
    q[1] = sin_half_alpha * cos_half_theta;
    q[2] = -sin_half_alpha * sin_half_theta;
    q[3] = cos_half_alpha * sin_half_theta;
    q[4] = -(a_sin_half_alpha * cos_half_theta + d_cos_half_alpha * sin_half_theta) / 2.0;
    q[5] =  (a_cos_half_alpha * cos_half_theta - d_sin_half_alpha * sin_half_theta) / 2.0;
    q[6] = -(a_cos_half_alpha * sin_half_theta + d_sin_half_alpha * cos_half_theta) / 2.0;
    q[7] =  (d_cos_half_alpha * cos_half_theta - a_sin_half_alpha * sin_half_theta) / 2.0;
}

template <typename scalar_t>
__device__ __forceinline__ void get_w_inline(const scalar_t* dh, const int ith, const int dh_type, scalar_t* w)
{
    if (dh_type == 1) { // Assuming dh_type 1 indicates 'mdh'
        scalar_t alpha = dh[3 + ith * 5];
        scalar_t a = dh[2 + ith * 5];
        w[0] = 0;
        w[1] = 0;
        w[2] = -sinf(alpha);
        w[3] = cosf(alpha);
        w[4] = 0;
        w[5] = 0;
        w[6] = -a * cosf(alpha);
        w[7] = -a * sinf(alpha);
    } else {
        w[0] = 0;
        w[1] = 0;
        w[2] = 0;
        w[3] = 1;
        w[4] = 0;
        w[5] = 0;
        w[6] = 0;
        w[7] = 0;
    }
} 

#define MAX_ITH 20  // Define an appropriate maximum value

template <typename scalar_t>
__device__ __forceinline__ void rel_abs_pose_rel_jac_inline(const scalar_t* dh1, const scalar_t* dh2,
                                                     const scalar_t* base1, const scalar_t* base2,
                                                     const scalar_t* effector1, const scalar_t* effector2,
                                                     const scalar_t* theta1,  const scalar_t* theta2,
                                                     const scalar_t* line_d, const scalar_t* quat_line_ref,
                                                     const int ith1, const int ith2,
                                                     scalar_t* rel_pose, scalar_t* abs_pose, scalar_t** Jxr, 
                                                     scalar_t* abs_position, scalar_t* angle, 
                                                     const int dh1_type, const int dh2_type)
{
    // Ensure ith1 and ith2 do not exceed MAX_ITH
    if (ith1 > MAX_ITH || ith2 > MAX_ITH) {
        // Handle error (e.g., return or set default values)
        return;
    }
    // temp for relative pose
    scalar_t x_effector1[8], x_effector2[8], x1[8], x2[8];
    scalar_t j1[8] = {0.0};
    scalar_t j2[8] = {0.0};
    scalar_t z1[8] = {0.0};
    scalar_t z2[8] = {0.0};
    scalar_t a1[8] = {0.0};
    scalar_t a2[8] = {0.0};
    scalar_t A1[8 * MAX_ITH], A2[8 * MAX_ITH];
    scalar_t J1[8 * MAX_ITH], J2[8 * MAX_ITH];
    scalar_t w[8] = {0, 0, 0, 1, 0, 0, 0, 0};
    scalar_t base1_hp8[64], base2_hp8[64], base1_hm8[64], base2_hm8[64];
    scalar_t effector1_hm8[64], effector2_hm8[64];
    scalar_t J1_temp1[64], J2_temp1[64];
    scalar_t J1_temp2[8 * MAX_ITH], J2_temp2[8 * MAX_ITH];
    scalar_t x_effector1_sqrt[8] = {0.0};
    scalar_t x_effector2_conj[8] = {0.0};
    scalar_t C8[64];
    scalar_t hp8_x_effector2_conj[64], hm8_x_effector1[64];
    scalar_t Jxr1_temp[8 * MAX_ITH], Jxr2_temp[8 * MAX_ITH], Jxr2_C8[64];
    scalar_t l_d_temp[4] = {0.0};
    scalar_t l_c[4] = {0.0};
    scalar_t abs_q[4] = {0.0};
    scalar_t abs_q_conj[4] = {0.0};

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++){
                C8[j + i*8] = 0;  // Initialize all elements to 0
        }
    }
    C8[0] =  1;
    C8[9] = -1;
    C8[18] = -1;
    C8[27] = -1;
    C8[36] =  1;
    C8[45] = -1;
    C8[54] = -1;
    C8[63] = -1;

    for (int i = 0; i < 8; i++) {
        x_effector1[i] = (i == 0) ? 1 : 0;
        x_effector2[i] = (i == 0) ? 1 : 0;
        x1[i] = (i == 0) ? 1 : 0;
        x2[i] = (i == 0) ? 1 : 0;
    }

    // get robot1 effector
    for(int i = 0; i < ith1; i++)
    {
        if (dh1_type == 1) { // Assuming mode 1 indicates 'mdh'
            mdh2dq_inline(dh1, theta1, i, a1); // Use mdh function
        } else {
            dh2dq_inline(dh1, theta1, i, a1); // Use standard DH function
        }
        dq_mult_inline(x_effector1, a1, x_effector1);
        for (int j = 0; j < 8; j++) {
            A1[j + i * 8] = a1[j];
        }
    }

    // get robot2 effector
    for(int i = 0; i < ith2; i++)
    {
        if (dh2_type == 1) { // Assuming mode 1 indicates 'mdh'
            mdh2dq_inline(dh2, theta2, i, a2); // Use mdh function
        } else {
            dh2dq_inline(dh2, theta2, i, a2); // Use standard DH function
        }
        dq_mult_inline(x_effector2, a2, x_effector2);
        for (int j = 0; j < 8; j++) {
            A2[j + i * 8] = a2[j];
        }
    }

    for(int i = 0; i < ith1; i++) {
        get_w_inline(dh1, i, dh1_type, w);
        Ad_inline(x1, w, z1);
        for (int j = 0; j < 8; j++) {
            z1[j] *= 0.5;
        }
        dq_mult_inline(x1, &A1[i * 8], x1);
        dq_mult_inline(z1, x_effector1, j1);
        
        for (int j = 0; j < 8; j++) {
            J1[j * ith1 + i] = j1[j];
        }
    }
    
    for (int i = 0; i < ith2; i++) {
        get_w_inline(dh2, i, dh2_type, w);
        Ad_inline(x2, w, z2);
        for (int j = 0; j < 8; j++) {
            z2[j] *= 0.5;
        }
        dq_mult_inline(x2, &A2[i * 8], x2);
        dq_mult_inline(z2, x_effector2, j2);
        for (int j = 0; j < 8; j++) {
            J2[j * ith2 + i] = j2[j];
        }
    }

    hamiplus8_inline(base1, base1_hp8);
    hamiplus8_inline(base2, base2_hp8);
    haminus8_inline(effector1, effector1_hm8);
    haminus8_inline(effector2, effector2_hm8);
    mat_mul88_inline(base1_hp8, effector1_hm8, J1_temp1);
    mat_mul88_inline(base2_hp8, effector2_hm8, J2_temp1);
    mat_mul_inline(J1_temp1, J1, J1_temp2, 8, 8, ith1);
    mat_mul_inline(J2_temp1, J2, J2_temp2, 8, 8, ith2);
    dq_mult_inline(base1, x_effector1, x_effector1);
    dq_mult_inline(base2, x_effector2, x_effector2);
    dq_mult_inline(x_effector1, effector1, x_effector1);
    dq_mult_inline(x_effector2, effector2, x_effector2);
    conj_inline(x_effector2, x_effector2_conj);
    
    dq_mult_inline(x_effector2_conj, x_effector1, rel_pose);
    dq_sqrt_inline(rel_pose, x_effector1_sqrt);
    dq_mult_inline(x_effector2, x_effector1_sqrt, abs_pose);
    hamiplus8_inline(x_effector2_conj, hp8_x_effector2_conj);
    haminus8_inline(x_effector1, hm8_x_effector1);
    mat_mul_inline(hp8_x_effector2_conj, J1_temp2, Jxr1_temp, 8, 8, ith1);
    mat_mul88_inline(hm8_x_effector1, C8, Jxr2_C8);
    mat_mul_inline(Jxr2_C8, J2_temp2, Jxr2_temp, 8, 8, ith2);
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < (ith1 + ith2); j++)
        {
            if(j < ith1)
            {Jxr[i][j] = Jxr1_temp[j + ith1 * i];}
            else
            {Jxr[i][j] = Jxr2_temp[j - ith1 + ith2 * i];}
        }
    }
    qconj_inline(abs_pose, abs_q_conj);
    q_mult_inline(abs_pose, line_d, l_d_temp);
    q_mult_inline(l_d_temp, abs_q_conj, l_c);
    q_angle_inline(l_c, quat_line_ref, angle);
    dq_position_inline(abs_pose, abs_position);
}

template <typename scalar_t>
__device__ __forceinline__ void rel_abs_pose_rel_abs_jac_inline(const scalar_t* dh1, const scalar_t* dh2,
                                                     const scalar_t* base1, const scalar_t* base2,
                                                     const scalar_t* effector1, const scalar_t* effector2,
                                                     const scalar_t* theta1,  const scalar_t* theta2,
                                                     const scalar_t* line_d, const scalar_t* quat_line_ref,
                                                     const int ith1, const int ith2,
                                                     scalar_t* rel_pose,  scalar_t* abs_pose, scalar_t** Jxr, scalar_t** Jxa,
                                                     scalar_t* abs_position, scalar_t* angle, 
                                                     const int dh1_type, const int dh2_type)
{
    // Ensure ith1 and ith2 do not exceed MAX_ITH
    if (ith1 > MAX_ITH || ith2 > MAX_ITH) {
        // Handle error (e.g., return or set default values)
        return;
    }

    scalar_t x_effector1[8], x_effector2[8], x1[8], x2[8];
    scalar_t j1[8] = {0.0};
    scalar_t j2[8] = {0.0};
    scalar_t z1[8] = {0.0};
    scalar_t z2[8] = {0.0};
    scalar_t a1[8] = {0.0};
    scalar_t a2[8] = {0.0};
    scalar_t A1[8 * MAX_ITH], A2[8 * MAX_ITH];
    scalar_t J1[8 * MAX_ITH], J2[8 * MAX_ITH];
    scalar_t w[8] = {0, 0, 0, 1, 0, 0, 0, 0};
    scalar_t base1_hp8[64], base2_hp8[64], base1_hm8[64], base2_hm8[64];
    scalar_t effector1_hm8[64], effector2_hm8[64];
    scalar_t J1_temp1[64], J2_temp1[64];
    scalar_t J1_temp2[8 * MAX_ITH], J2_temp2[8 * MAX_ITH];
    scalar_t x_effector1_sqrt[8] = {0.0};
    scalar_t x_effector2_conj[8] = {0.0};
    scalar_t C8[64];
    scalar_t C4[16];
    scalar_t hp8_x_effector2_conj[64], hm8_x_effector1[64];
    scalar_t Jxr1_temp[8 * MAX_ITH], Jxr2_temp[8 * MAX_ITH], Jxr2_C8[64];
    scalar_t l_d_temp[4] = {0.0};
    scalar_t l_c[4] = {0.0};
    scalar_t abs_q[4] = {0.0};
    scalar_t abs_q_conj[4] = {0.0};

    // temp for absolute jacobian
    // init C8 and C4
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++){
                C8[j + i*8] = 0;  // Initialize all elements to 0
        }
    }
    C8[0] =  1;
    C8[9] = -1;
    C8[18] = -1;
    C8[27] = -1;
    C8[36] =  1;
    C8[45] = -1;
    C8[54] = -1;
    C8[63] = -1;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++){
                C4[j + i*4] = 0;  // Initialize all elements to 0
        }
    }
    C4[0] =  1;
    C4[5] = -1;
    C4[10] = -1;
    C4[15] = -1;
    // init x_effector1, x_effector2, x1, x2
    for (int i = 0; i < 8; i++) {
        x_effector1[i] = (i == 0) ? 1 : 0;
        x_effector2[i] = (i == 0) ? 1 : 0;
        x1[i] = (i == 0) ? 1 : 0;
        x2[i] = (i == 0) ? 1 : 0;
    }

    // get robot1 effector
    for(int i = 0; i < ith1; i++)
    {
        if (dh1_type == 1) { // Assuming mode 1 indicates 'mdh'
            mdh2dq_inline(dh1, theta1, i, a1); // Use mdh function
        } else {
            dh2dq_inline(dh1, theta1, i, a1); // Use standard DH function
        }
        dq_mult_inline(x_effector1, a1, x_effector1);
        for (int j = 0; j < 8; j++) {
            A1[j + i * 8] = a1[j];
        }
    }

    // get robot2 effector
    for(int i = 0; i < ith2; i++)
    {
        if (dh2_type == 1) { // Assuming mode 1 indicates 'mdh'
            mdh2dq_inline(dh2, theta2, i, a2); // Use mdh function
        } else {
            dh2dq_inline(dh2, theta2, i, a2); // Use standard DH function
        }
        dq_mult_inline(x_effector2, a2, x_effector2);
        for (int j = 0; j < 8; j++) {
            A2[j + i * 8] = a2[j];
        }
    }

    for(int i = 0; i < ith1; i++) {
        get_w_inline(dh1, i, dh1_type, w);
        Ad_inline(x1, w, z1);
        for (int j = 0; j < 8; j++) {
            z1[j] *= 0.5;
        }
        dq_mult_inline(x1, &A1[i * 8], x1);
        dq_mult_inline(z1, x_effector1, j1);
        
        for (int j = 0; j < 8; j++) {
            J1[j * ith1 + i] = j1[j];
        }
    }
    
    for (int i = 0; i < ith2; i++) {
        get_w_inline(dh2, i, dh2_type, w);
        Ad_inline(x2, w, z2);
        for (int j = 0; j < 8; j++) {
            z2[j] *= 0.5;
        }
        dq_mult_inline(x2, &A2[i * 8], x2);
        dq_mult_inline(z2, x_effector2, j2);
        for (int j = 0; j < 8; j++) {
            J2[j * ith2 + i] = j2[j];
        }
    }

    hamiplus8_inline(base1, base1_hp8);
    hamiplus8_inline(base2, base2_hp8);
    haminus8_inline(effector1, effector1_hm8);
    haminus8_inline(effector2, effector2_hm8);
    mat_mul88_inline(base1_hp8, effector1_hm8, J1_temp1);
    mat_mul88_inline(base2_hp8, effector2_hm8, J2_temp1);
    mat_mul_inline(J1_temp1, J1, J1_temp2, 8, 8, ith1);
    mat_mul_inline(J2_temp1, J2, J2_temp2, 8, 8, ith2);
    dq_mult_inline(base1, x_effector1, x_effector1);
    dq_mult_inline(base2, x_effector2, x_effector2);
    dq_mult_inline(x_effector1, effector1, x_effector1);
    dq_mult_inline(x_effector2, effector2, x_effector2);
    conj_inline(x_effector2, x_effector2_conj);
    
    dq_mult_inline(x_effector2_conj, x_effector1, rel_pose);
    dq_sqrt_inline(rel_pose, x_effector1_sqrt);
    dq_mult_inline(x_effector2, x_effector1_sqrt, abs_pose);
    hamiplus8_inline(x_effector2_conj, hp8_x_effector2_conj);
    haminus8_inline(x_effector1, hm8_x_effector1);
    mat_mul_inline(hp8_x_effector2_conj, J1_temp2, Jxr1_temp, 8, 8, ith1);
    mat_mul88_inline(hm8_x_effector1, C8, Jxr2_C8);
    mat_mul_inline(Jxr2_C8, J2_temp2, Jxr2_temp, 8, 8, ith2);
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < (ith1 + ith2); j++)
        {
            if(j < ith1)
            {Jxr[i][j] = Jxr1_temp[j + ith1 * i];}
            else
            {Jxr[i][j] = Jxr2_temp[j - ith1 + ith2 * i];}
        }
    }
    qconj_inline(abs_pose, abs_q_conj);
    q_mult_inline(abs_pose, line_d, l_d_temp);
    q_mult_inline(l_d_temp, abs_q_conj, l_c);
    q_angle_inline(l_c, quat_line_ref, angle);
    dq_position_inline(abs_pose, abs_position);

    // temp for rotation_jacobian
    scalar_t Jxr_temp[8 * 2* MAX_ITH] = {0.0};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < (ith1 + ith2); j++){
            Jxr_temp[j + i*(ith1+ith2)] = Jxr[i][j];  // Initialize all elements to 0
        }
    }
    scalar_t Jxr_rotation[8 * MAX_ITH]= {0.0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < (ith1+ith2); j++){
            Jxr_rotation[j + i*(ith1+ith2)] = Jxr_temp[j + i*(ith1+ith2)];  // Initialize all elements to 0
        }
    }
    // temp1 for translation_jacobian
    scalar_t xr[8]= {0.0};
    for (int i = 0; i < 8; i++) {
        xr[i] = rel_pose[i];
    }
    scalar_t xr_p[8]= {0.0} ;
    scalar_t xr_p_conj[8]= {0.0};
    scalar_t xr_p_conj_hm4[16]= {0.0};
    scalar_t Jxr_block[8 * MAX_ITH]= {0.0};
    scalar_t trans_temp1[8 * MAX_ITH]= {0.0};
    // temp2 for translation_jacobian
    scalar_t xr_d[8]= {0.0};
    scalar_t xr_d_hp4[16]= {0.0};
    scalar_t trans_temp2_C4[16]= {0.0};
    scalar_t trans_temp2[8 * MAX_ITH]= {0.0};
    scalar_t Jxr_trans[8 * MAX_ITH]= {0.0};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < (ith1 + ith2); j++) {
        Jxr_block[i * (ith1 + ith2) + j] = Jxr_temp[(i + 4) * (ith1 + ith2) + j];
    }
    }
    // calculate for jxr_trans
    P_inline(xr,xr_p);
    conj_inline(xr_p, xr_p_conj);
    haminus4_inline(xr_p_conj, xr_p_conj_hm4);
    mat_mul_inline(xr_p_conj_hm4, Jxr_block, trans_temp1, 4, 4, (ith1 + ith2));
    D_inline(xr, xr_d);
    hamiplus4_inline(xr_d, xr_d_hp4);
    mat_mul_inline(xr_d_hp4, C4, trans_temp2_C4, 4, 4, 4);
    mat_mul_inline(trans_temp2_C4, Jxr_rotation, trans_temp2, 4, 4, (ith1 + ith2));
     for (int i = 0; i < 4; i++) {
        for (int j = 0; j < (ith1 + ith2); j++) {
        Jxr_trans[i * (ith1 + ith2) + j] = 2.0*(trans_temp1[i * (ith1 + ith2) + j] + trans_temp2[i * (ith1 + ith2) + j]);
    }
    }
    // calculate for Jrr2
    scalar_t Jrr2[8*MAX_ITH]= {0.0};
    scalar_t xr_p_pow[8]= {0.0};
    scalar_t Jrr2_temp1[8]= {0.0};
    scalar_t Jrr2_temp2[8]= {0.0};
    scalar_t Jrr2_temp1_hm4[16]= {0.0};
    //conj(xr.P())*pow(xr.P(),0.5)
    dq_sqrt_inline(xr_p, xr_p_pow);
    dq_mult_inline(xr_p_conj, xr_p_pow, Jrr2_temp1);
    haminus4_inline(Jrr2_temp1, Jrr2_temp1_hm4);
    mat_mul_inline(Jrr2_temp1_hm4, Jxr_rotation, Jrr2_temp2, 4, 4, (ith1 + ith2));
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < (ith1 + ith2); j++) {
        Jrr2[i * (ith1 + ith2) + j] = 0.5*Jrr2_temp2[i * (ith1 + ith2) + j];
    }
    }
    // calculate for Jxr2
    scalar_t Jxr2[8 * 2* MAX_ITH]= {0.0};
    scalar_t xr_p_pow_hm4[16]= {0.0};
    scalar_t xr_trans[8]= {0.0};
    scalar_t xr_trans_hp4[16]= {0.0};
    scalar_t Jxr2_temp1[8*MAX_ITH]= {0.0};
    scalar_t Jxr2_temp2[8*MAX_ITH]= {0.0};
    haminus4_inline(xr_p_pow, xr_p_pow_hm4);
    mat_mul_inline(xr_p_pow_hm4, Jxr_trans, Jxr2_temp1, 4, 4, ith1 + ith2);
    translation_inline(xr, xr_trans);
    hamiplus4_inline(xr_trans, xr_trans_hp4);
    mat_mul_inline(xr_trans_hp4, Jrr2, Jxr2_temp2, 4, 4, ith1 + ith2);
    for (int i = 0; i < 8; i++) 
    {
        if (i < 4) 
        {
        for (int j = 0; j < (ith1 + ith2); j++)
        {
            Jxr2[i * (ith1 + ith2) + j] = Jrr2[i * (ith1 + ith2) + j];
        }
        } 
        else 
        {
        for (int j = 0; j < (ith1 + ith2); j++) {
            Jxr2[i * (ith1 + ith2) + j] = 0.25*Jxr2_temp1[(i - 4) * (ith1 + ith2) + j] + Jxr2_temp2[(i - 4) * (ith1 + ith2) + j];
        }
        }
    }
    // calculate for Jxr2 temp1
    scalar_t Jxa_temp1[8 * 2* MAX_ITH]= {0.0};
    scalar_t Jxa_temp2[8 * 2* MAX_ITH]= {0.0};
    scalar_t Jxa_temp3[8 * 2* MAX_ITH]= {0.0};
    for (int i = 0; i < 8; i++) {
        for (int j = ith1; j < (ith1+ ith2); j++) {
            Jxa_temp1[i * (ith1 + ith2) + j] = J2[i * (ith2) + (j-ith1)];
        }
    }
    scalar_t xr_pow[8]= {0.0};
    scalar_t xr_pow_hm8[64]= {0.0};
    dq_sqrt_inline(xr, xr_pow);
    haminus8_inline(xr_pow, xr_pow_hm8);
    mat_mul_inline(xr_pow_hm8, Jxa_temp1,Jxa_temp2, 8, 8, (ith1 + ith2));
    scalar_t x2_hp8[64]= {0.0};
    hamiplus8_inline(x2, x2_hp8);
    mat_mul_inline(x2_hp8, Jxr2, Jxa_temp3, 8, 8, (ith1 + ith2));
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < (ith1 + ith2); j++)
        {
            Jxa[i][j] = Jxa_temp2[i * (ith1 + ith2) + j] + Jxa_temp3[i * (ith1 + ith2) + j];
        }
    }
}

//----------------------------- kernel 函数------------------------------------
//----------------------------- kernel 函数------------------------------------
//----------------------------- kernel 函数------------------------------------
template <typename scalar_t>
__global__ void dq_mult_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> q1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> q2,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> results
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < q1.size(0)) {
        // 提取输入四元数的实部和双部
        scalar_t a1 = q1[idx][0];
        scalar_t b1 = q1[idx][1];
        scalar_t c1 = q1[idx][2];
        scalar_t d1 = q1[idx][3];
        scalar_t ad1 = q1[idx][4];
        scalar_t bd1 = q1[idx][5];
        scalar_t cd1 = q1[idx][6];
        scalar_t dd1 = q1[idx][7];

        scalar_t a2 = q2[idx][0];
        scalar_t b2 = q2[idx][1];
        scalar_t c2 = q2[idx][2];
        scalar_t d2 = q2[idx][3];
        scalar_t ad2 = q2[idx][4];
        scalar_t bd2 = q2[idx][5];
        scalar_t cd2 = q2[idx][6];
        scalar_t dd2 = q2[idx][7];

        // 计算实部四元数乘法
        results[idx][0] = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2;
        results[idx][1] = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2;
        results[idx][2] = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2;
        results[idx][3] = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2;

        // 计算双部四元数乘法
        results[idx][4] = ad1 * a2 + a1 * ad2 - bd1 * b2 - b1 * bd2 - cd1 * c2 - c1 * cd2 - dd1 * d2 - d1 * dd2;
        results[idx][5] = ad1 * b2 + a1 * bd2 + bd1 * a2 + b1 * ad2 + cd1 * d2 + c1 * dd2 - dd1 * c2 - d1 * cd2;
        results[idx][6] = ad1 * c2 + a1 * cd2 - bd1 * d2 - b1 * dd2 + cd1 * a2 + c1 * ad2 + dd1 * b2 + d1 * bd2;
        results[idx][7] = ad1 * d2 + a1 * dd2 + bd1 * c2 + b1 * cd2 - cd1 * b2 - c1 * bd2 + dd1 * a2 + d1 * ad2;
    }
}

template <typename scalar_t>
__global__ void P_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> new_q,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 8; i++) {
            new_q[idx][i] = (i < 4) ? v[idx][i] : 0;
        }
    }
}

template <typename scalar_t>
__global__ void D_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> new_q,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 8; i++) {
            new_q[idx][i] = (i < 4) ? v[idx][i + 4] : 0;
        }
    }
}

template <typename scalar_t>
__global__ void Re_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> new_q,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 8; i++) {
            new_q[idx][i] = 0;
        }
        new_q[idx][0] = v[idx][0];
        new_q[idx][4] = v[idx][4];
    }
}

template <typename scalar_t>
__global__ void Im_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> new_q,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        for (int i = 0; i < 8; i++) {
            new_q[idx][i] = ((i > 0 && i < 4) || i > 4) ? v[idx][i] : 0;
        }
    }
}

template <typename scalar_t>
__global__ void conj_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> conj_q,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        conj_q[idx][0] = v[idx][0];
        conj_q[idx][1] = -v[idx][1];
        conj_q[idx][2] = -v[idx][2];
        conj_q[idx][3] = -v[idx][3];
        // 双四元数的第二部分也需要处理
        conj_q[idx][4] = v[idx][4];
        conj_q[idx][5] = -v[idx][5];
        conj_q[idx][6] = -v[idx][6];
        conj_q[idx][7] = -v[idx][7];
    }
}

template <typename scalar_t>
__global__ void translation_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> translation_result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < v.size(0)) {
        scalar_t p[8], d[8], conj_p[8], result[8];

        // 初始化 P(v)
        p[0] = v[idx][0];
        p[1] = v[idx][1];
        p[2] = v[idx][2];
        p[3] = v[idx][3];
        for (int i = 4; i < 8; i++) {
            p[i] = 0;
        }

        // 初始化 D(v)
        for (int i = 0; i < 4; i++) {
            d[i] = v[idx][i+4];
        }
        for (int i = 4; i < 8; i++) {
            d[i] = 0;
        }

        // 计算 P(v) 的共轭
        conj_inline(p, conj_p);

        // 双四元数乘法
        dq_mult_inline(d, conj_p, result);

        // 存储结果
        for (int i = 0; i < 8; i++) {
            translation_result[idx][i] = 2.0 * result[i];
        }
    }
}

template <typename scalar_t>
__global__ void rotation_angle_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> angle,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        scalar_t real_part = v[idx][0];
        real_part = max(min(real_part, scalar_t(1.0)), scalar_t(-1.0));  // Clamp to [-1, 1]
        angle[idx] = 2 * acos(real_part);
}
}
template <typename scalar_t>
__global__ void norm_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> norm,
    int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        scalar_t conj_v[8];
        conj_v[0] = v[idx][0];
        for (int i = 1; i < 8; i++) {
            conj_v[i] = -v[idx][i];
        }

        scalar_t result[8];
        dq_mult_inline(conj_v, v[idx].data(), result);  // 使用 __device__ 函数

        scalar_t real_part = result[0];
        scalar_t dual_part = result[4];

        norm[idx][0] = sqrt(real_part);
        norm[idx][4] = dual_part / (2 * norm[idx][0]);
    }
}

template <typename scalar_t>
__global__ void rotation_axis_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> results,
    int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        scalar_t q[8];
        scalar_t im_q[8];
        scalar_t phi_tensor[8];
        scalar_t rot_axis[8];

        // 复制双四元数
        for (int i = 0; i < 8; i++) {
            q[i] = v[idx][i];
        }

        // 计算旋转角度
        scalar_t phi = rotation_angle_inline(q) / 2.0;
        scalar_t sin_phi = sin(phi);
        for (int i = 0; i < 8; i++) {
            phi_tensor[i] = 0;
        }
        phi_tensor[0] = 1 / sin_phi; // 只有实部不为0

        // 获取虚部
        Im_inline(q, im_q);

        // 双四元数乘法
        dq_mult_inline(phi_tensor, im_q, rot_axis);

        // 存储结果
        for (int i = 0; i < 8; i++) {
            results[idx][i] = rot_axis[i];
        }
    }
}

template <typename scalar_t>
__global__ void dq_log_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> results,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        scalar_t phi_tensor[8] = {0.0};
        scalar_t rot_axis[8] = {0.0};
        scalar_t p[8]  ={0.0};
        scalar_t d[8]  ={0.0};
        scalar_t pv[8] ={0.0};
        // rotation_axis(v)
        scalar_t phi = 0.5 *rotation_angle_inline(&v[idx][0]);
        phi_tensor[0] = phi;
        rotation_axis_inline(&v[idx][0], rot_axis);
        dq_mult_inline(phi_tensor, rot_axis, p);
        translation_inline(&v[idx][0], d);
        for (int i = 0; i < 8; i++) { // 只处理平移部分
            d[i] *= 0.5;
        }
        for (int i = 0; i < 4; i++) {
            results[idx][i] = p[i];
        }
        for (int i = 4; i < 8; i++) {
            results[idx][i] = d[i-4];
        }

    }
}

template <typename scalar_t>
__global__ void dq_exp_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> results,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        scalar_t phi_tensor[8] = {0.0};
        scalar_t prim[8] = {0.0};
        scalar_t dual[8] = {0.0};
        scalar_t new_prim[8] ={0.0};
        scalar_t temp[8] ={0.0};
        scalar_t d[8] ={0.0};
        scalar_t E_[8] ={0.0};
        scalar_t phi = 0.0;
        E_[4] = 1.0;
        P_inline(&v[idx][0], prim);
        for (int i = 0; i < 8; i++) {
            phi += pow(prim[i], 2);
        }
        phi = sqrt(phi);
        phi_tensor[0] = sin(phi)/(phi+ 1e-8);
        dq_mult_inline(phi_tensor, prim, new_prim);
        new_prim[0] += cos(phi);
        D_inline(&v[idx][0], d);
        dq_mult_inline(E_, d, temp);
        dq_mult_inline(temp, new_prim, dual);
        for(int i = 0; i < 8; i++) {
            results[idx][i] = dual[i]+new_prim[i];
        }
    }
}

template <typename scalar_t>
__global__ void dq_sqrt_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> results,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) 
    {
        scalar_t dq_log_value[8] = {0.0};
        dq_log_inline(&v[idx][0], dq_log_value);
        for (int i = 0; i < 8; i++) {
            dq_log_value[i] *= 0.5;
        }
        dq_exp_inline(dq_log_value, &results[idx][0]);
    }
}

template <typename scalar_t>
__global__ void dq_inv_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> results,
    int N)
    {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dq_inv_inline(&v[idx][0], &results[idx][0]);
    }
}

template <typename scalar_t>
__global__ void dq_normalize_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> results,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dq_normalize_inline(&v[idx][0], &results[idx][0]);
    }
}

template <typename scalar_t>
__global__ void Ad_kernel(    
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> q1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> q2,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> results,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Ad_inline(&q1[idx][0], &q2[idx][0], &results[idx][0]);
    }
}

template <typename scalar_t>
__global__ void hamiplus8_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> results,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        scalar_t* result_ptr[8];
        for (int i = 0; i < 8; ++i) {
            result_ptr[i] = &results[idx][i][0];
        }
        hamiplus8_inline_v0(&v[idx][0], result_ptr);
    }
}

template <typename scalar_t>
__global__ void haminus8_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> v,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> results,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        scalar_t* result_ptr[8];
        for (int i = 0; i < 8; ++i) {
            result_ptr[i] = &results[idx][i][0];
        }
        haminus8_inline_v0(&v[idx][0], result_ptr);
    }
}



// const scalar_t** dh1, const scalar_t** dh2,
// const scalar_t* base1, const scalar_t* base2,
// const scalar_t* effector1, const scalar_t* effector2,
// scalar_t theta1, scalar_t theta2,
// const int ith, scalar_t* rel_pose, scalar_t* abs_pose, scalar_t** Jxr)
template <typename scalar_t>
__global__ void rel_abs_pose_rel_jac_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dh1,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dh2,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> base1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> base2,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> effector1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> effector2,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> theta1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> theta2,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> line_d,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> quat_line_ref,
    int ith1, int ith2, int N,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rel_pose,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> abs_pose,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> Jxr,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> abs_position,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> angle,
    int dh1_type, int dh2_type)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < N) {
    // 准备Jxr_ptr
    scalar_t* Jxr_ptr[8];
    for (int i = 0; i < 8; i++) {
        Jxr_ptr[i] = &Jxr[idx][i][0];
    }
    rel_abs_pose_rel_jac_inline(
        &dh1[0], &dh2[0],
        &base1[idx][0], &base2[idx][0],
        &effector1[idx][0], &effector2[idx][0],
        &theta1[idx][0], &theta2[idx][0],
        &line_d[idx][0], &quat_line_ref[idx][0],
        ith1, ith2,
        &rel_pose[idx][0], &abs_pose[idx][0], Jxr_ptr, &abs_position[idx][0], &angle[idx][0], dh1_type, dh2_type);
    }
}
    
    
template <typename scalar_t>
__global__ void rel_abs_pose_rel_abs_jac_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dh1,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dh2,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> base1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> base2,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> effector1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> effector2,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> theta1,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> theta2,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> line_d,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> quat_line_ref,
    int ith1, int ith2, int N,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rel_pose,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> abs_pose,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> Jxr,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> Jxa,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> abs_position,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> angle,
    int dh1_type, int dh2_type)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < N) {
    // 准备Jxr_ptr
    scalar_t* Jxr_ptr[8];
    scalar_t* Jxa_ptr[8];
    for (int i = 0; i < 8; i++) {
        Jxr_ptr[i] = &Jxr[idx][i][0];
        Jxa_ptr[i] = &Jxa[idx][i][0];
    }
    rel_abs_pose_rel_abs_jac_inline(
        &dh1[0], &dh2[0],
        &base1[idx][0], &base2[idx][0],
        &effector1[idx][0], &effector2[idx][0],
        &theta1[idx][0], &theta2[idx][0],
        &line_d[idx][0], &quat_line_ref[idx][0],
        ith1, ith2,
        &rel_pose[idx][0], &abs_pose[idx][0], Jxr_ptr, Jxa_ptr, &abs_position[idx][0], &angle[idx][0], dh1_type, dh2_type);
    }
}


//----------------------------- cuda 函数------------------------------------
//----------------------------- cuda 函数------------------------------------
//----------------------------- cuda 函数------------------------------------

torch::Tensor dq_mult_cuda(torch::Tensor q1, torch::Tensor q2)
{
    const int N = q1.size(0);
    torch::Tensor results = torch::zeros({N, 8}, q1.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(q1.type(), "dq_mult_kernel", 
    ([&] {
            dq_mult_kernel<scalar_t><<<blocks, threads>>>(
            q1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            q2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));
    // cudaDeviceSynchronize();
    return results; 
}

torch::Tensor P_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor new_q = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "P_kernel", 
    ([&] {
            P_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            new_q.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return new_q; 
}

torch::Tensor D_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor new_q = torch::zeros({N, 8}, v.options());
    const int threads =256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "D_kernel", 
    ([&] {
            D_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            new_q.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return new_q; 
}

torch::Tensor Re_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor new_q = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "Re_kernel", 
    ([&] {
            Re_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            new_q.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return new_q; 
}

torch::Tensor Im_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor new_q = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "Im_kernel", 
    ([&] {
            Im_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            new_q.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return new_q; 
}

torch::Tensor conj_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor conj_q = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "conj_kernel", 
    ([&] {
            conj_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            conj_q.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return conj_q; 
}

torch::Tensor norm_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor norm = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "norm_kernel", 
    ([&] {
            norm_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            norm.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return norm; 
}

torch::Tensor translation_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor translation_result = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "translation_kernel", 
    ([&] {
            translation_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            translation_result.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));
    // cudaDeviceSynchronize();
    return translation_result; 
}

torch::Tensor rotation_angle_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor angle = torch::zeros({N}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "rotation_angle_kernel", 
    ([&] {
            rotation_angle_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            angle.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return angle; 
}

torch::Tensor rotation_axis_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor results = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "rotation_axis_kernel", 
    ([&] {
            rotation_axis_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return results; 
}

torch::Tensor dq_log_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor results = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "dq_log_kernel", 
    ([&] {
            dq_log_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return results; 
}

torch::Tensor dq_exp_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor results = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "dq_exp_kernel", 
    ([&] {
            dq_exp_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return results; 
}

torch::Tensor dq_sqrt_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor results = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "dq_sqrt_kernel", 
    ([&] {
            dq_sqrt_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return results; 
}

torch::Tensor dq_inv_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor results = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "dq_inv_kernel", 
    ([&] {
            dq_inv_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return results; 
}

torch::Tensor dq_normalize_cuda(torch::Tensor v)
{
    const int N = v.size(0);
    torch::Tensor results = torch::zeros({N, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "dq_normalize_kernel", 
    ([&] {
            dq_normalize_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return results; 
}

torch::Tensor Ad_cuda(torch::Tensor q1, torch::Tensor q2)
{
    const int N = q1.size(0);
    torch::Tensor results = torch::zeros({N, 8}, q1.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(q1.type(), "Ad_kernel", 
    ([&] {
            Ad_kernel<scalar_t><<<blocks, threads>>>(
            q1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            q2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize();
    return results; 
}

torch::Tensor hamiplus8_cuda(torch::Tensor v) {
    const int N = v.size(0);
    // 注意这里的维度是 {N, 8, 8}，因为我们需要存储8x8的矩阵
    torch::Tensor results = torch::zeros({N, 8, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "hamiplus_kernel", 
    ([&] {
        hamiplus8_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize(); // Uncomment if you encounter issues with synchronization
    return results; 
}

torch::Tensor haminus8_cuda(torch::Tensor v) {
    const int N = v.size(0);
    // 注意这里的维度是 {N, 8, 8}，因为我们需要存储8x8的矩阵
    torch::Tensor results = torch::zeros({N, 8, 8}, v.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(v.type(), "haminus8_kernel", 
    ([&] {
        haminus8_kernel<scalar_t><<<blocks, threads>>>(
            v.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            results.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            N
        );
    }));
    // cudaDeviceSynchronize(); // Uncomment if you encounter issues with synchronization
    return results; 
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_abs_pose_rel_jac_cuda
(torch::Tensor dh1, torch::Tensor dh2,
torch::Tensor base1, torch::Tensor base2,
torch::Tensor effector1, torch::Tensor effector2,
torch::Tensor theta1, torch::Tensor theta2,
torch::Tensor line_d, torch::Tensor quat_line_ref,
int ith1, int ith2, int dh1_type, int dh2_type)
{
    const int N = theta1.size(0);
    torch::Tensor rel_results = torch::zeros({N, 8}, theta1.options());
    torch::Tensor abs_results = torch::zeros({N, 8}, theta1.options());
    torch::Tensor jxr_results = torch::zeros({N, 8, (ith1+ith2)}, theta1.options());
    torch::Tensor abs_position_results = torch::zeros({N, 3}, theta1.options());
    torch::Tensor angle = torch::zeros({N, 1}, theta1.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(theta1.type(), "rel_abs_pose_rel_jac_kernel", 
    ([&] {
          rel_abs_pose_rel_jac_kernel<scalar_t><<<blocks, threads>>>(
          dh1.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
          dh2.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
          base1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          base2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          effector1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          effector2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          theta1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          theta2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), 
          line_d.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          quat_line_ref.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          ith1, ith2,
          N,
          rel_results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          abs_results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          jxr_results.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
          abs_position_results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          angle.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          dh1_type, dh2_type
        );
    }));
    cudaDeviceSynchronize();
    return std::make_tuple(rel_results, abs_results, jxr_results, abs_position_results, angle);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rel_abs_pose_rel_abs_jac_cuda
(torch::Tensor dh1, torch::Tensor dh2,
torch::Tensor base1, torch::Tensor base2,
torch::Tensor effector1, torch::Tensor effector2,
torch::Tensor theta1, torch::Tensor theta2,
torch::Tensor line_d, torch::Tensor quat_line_ref,
int ith1, int ith2, int dh1_type, int dh2_type)
{
    const int N = theta1.size(0);
    torch::Tensor rel_results = torch::zeros({N, 8}, theta1.options());
    torch::Tensor abs_results = torch::zeros({N, 8}, theta1.options());
    torch::Tensor jxr_results = torch::zeros({N, 8, (ith1+ith2)}, theta1.options());
    torch::Tensor jxa_results = torch::zeros({N, 8, (ith1+ith2)}, theta1.options());
    torch::Tensor abs_position_results = torch::zeros({N, 3}, theta1.options());
    torch::Tensor angle = torch::zeros({N, 1}, theta1.options());
    const int threads = 256; // 根据实际情况调整
    const dim3 blocks((N + threads - 1) / threads);
    AT_DISPATCH_FLOATING_TYPES(theta1.type(), "rel_abs_pose_rel_abs_jac_kernel", 
    ([&] {
          rel_abs_pose_rel_abs_jac_kernel<scalar_t><<<blocks, threads>>>(
          dh1.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
          dh2.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
          base1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          base2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          effector1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          effector2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          theta1.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          theta2.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(), 
          line_d.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          quat_line_ref.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          ith1, ith2,
          N,
          rel_results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          abs_results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          jxr_results.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
          jxa_results.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
          abs_position_results.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          angle.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
          dh1_type, dh2_type
        );
    }));
    cudaDeviceSynchronize();
    return std::make_tuple(rel_results, abs_results, jxr_results, jxa_results, abs_position_results, angle);
}