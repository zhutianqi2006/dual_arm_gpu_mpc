#pragma once
// dq_fast_projection.cuh
// Fast reordering for project_q_by_constraint_jacobian_inline.
// Author: ChatGPT (optimized path based on left-multiply-then-dot trick)
//
// Drop-in helper you can include and call from your kernel/inline function.
// It avoids building large 8xJ temporaries (J1/J2/Jxr/N) and replaces them
// with two 8x8^T·vec transforms plus J many 8-dim dot products.
//
// Requirements: The following device helpers must already exist in your codebase:
//   - hamiplus8_inline(const T* dq, T* out64)        // 8x8
//   - haminus8_inline(const T* dq, T* out64)         // 8x8
//   - mat_mul88_inline(const T* A, const T* B, T* C) // 8x8 = 8x8 * 8x8
//   - dq_mult_inline(const T* a, const T* b, T* out) // dual quaternion multiply (8)
//   - conj_inline(const T* dq, T* out)               // conjugate (8)
//   - Ad_inline(const T* x, const T* w, T* out)      // Adjoint on twist in dq algebra (8)
//   - get_w_inline(DH, i, dh_type, T* w)             // joint screw/twist (8)
//   - dh2dq_inline(DH, const T* theta, int i, T* a)  // product-of-exponentials (link i)
//   - mdh2dq_inline(DH, const T* theta, int i, T* a) // (if dh_type == 1)
//
// Template parameters DH1/DH2 are your DH container types (forwarded to helpers).
//
// Usage example (inside your existing function/kernel):
//   #include "dq_fast_projection.cuh"
//   project_q_by_constraint_jacobian_inline_fast(
//       joint_vel_dir, dq_tmp, desire_rel_pose,
//       base1, base2, effector1, effector2,
//       dh1, dh2, ith1, ith2, dh1_type, dh2_type,
//       theta1_local, theta2_local);
//
// If you want to alias the original slow name to this fast one:
//   #define project_q_by_constraint_jacobian_inline project_q_by_constraint_jacobian_inline_fast
//
// Notes:
// - This version computes x_eff1/x_eff2 in one pass, then recomputes per-link a_i
//   during the Jacobian column loop to build prefix cumulatively (O(J), not O(J^2)).
// - C8/D8 diagonal signs are applied with explicit sign flips (no memory loads).
// - Everything is kept in registers to lower local memory pressure.
//
// -----------------------------------------------------------------------------

#ifndef DQ_FAST_PROJECTION_CUH_
#define DQ_FAST_PROJECTION_CUH_

#if !defined(__CUDACC__)
#error "This header is intended for CUDA device code."
#endif

// ---- Small helpers ---------------------------------------------------------

template <typename T>
__device__ __forceinline__ T dq_dot8(const T* __restrict__ a, const T* __restrict__ b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] +
           a[4]*b[4] + a[5]*b[5] + a[6]*b[6] + a[7]*b[7];
}

// y = A^T * x, where A is 8x8, row-major (same layout as your hamiplus/haminus)
template <typename T>
__device__ __forceinline__ void dq_mul8T_vec(const T* __restrict__ A, const T* __restrict__ x, T* __restrict__ y) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        T acc = 0;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            acc += A[c*8 + r] * x[c];  // (A^T)[r,c] * x[c]
        }
        y[r] = acc;
    }
}

// y = D8 * x  with D8 = diag(+1,-1,-1,-1, +1,-1,-1,-1)
template <typename T>
__device__ __forceinline__ void dq_apply_D8(const T* __restrict__ x, T* __restrict__ y) {
    y[0] =  x[0]; y[1] = -x[1]; y[2] = -x[2]; y[3] = -x[3];
    y[4] =  x[4]; y[5] = -x[5]; y[6] = -x[6]; y[7] = -x[7];
}

// y = C8 * x  with C8 = diag(+1,-1,-1,-1, +1,-1,-1,-1)
template <typename T>
__device__ __forceinline__ void dq_apply_C8(const T* __restrict__ x, T* __restrict__ y) {
    // Same pattern as D8 in this context.
    dq_apply_D8(x, y);
}

// Identity dual quaternion (1,0,0,0, 0,0,0,0)
template <typename T>
__device__ __forceinline__ void dq_identity(T* __restrict__ q) {
    q[0]=T(1); q[1]=q[2]=q[3]=q[4]=q[5]=q[6]=q[7]=T(0);
}

// Copy 8
template <typename T>
__device__ __forceinline__ void dq_copy8(const T* __restrict__ a, T* __restrict__ b) {
#pragma unroll
    for (int i=0;i<8;++i) b[i]=a[i];
}

// ---- Core fast path --------------------------------------------------------

template <typename T, typename DH1, typename DH2>
__device__ __forceinline__
void project_q_by_constraint_jacobian_inline_fast(
    T* __restrict__ joint_vel_dir,                      // out: (ith1+ith2)
    const T* __restrict__ dq_tmp,                       // in : 8
    const T* __restrict__ desire_rel_pose,              // in : 8 (desired relative dq)
    const T* __restrict__ base1, const T* __restrict__ base2,           // 8, 8
    const T* __restrict__ effector1, const T* __restrict__ effector2,   // 8, 8
    const DH1& dh1, const DH2& dh2,
    const int ith1, const int ith2,
    const int dh1_type, const int dh2_type,
    const T* __restrict__ theta1_local,                 // ith1
    const T* __restrict__ theta2_local                  // ith2
) {
    // --- Precompute base/effector transforms --------------------------------
    T base1_hp8[64], base2_hp8[64];
    T eff1_hm8[64], eff2_hm8[64];
    hamiplus8_inline(base1, base1_hp8);
    hamiplus8_inline(base2, base2_hp8);
    haminus8_inline(effector1, eff1_hm8);
    haminus8_inline(effector2, eff2_hm8);

    T M1[64], M2[64];
    mat_mul88_inline(base1_hp8, eff1_hm8, M1);
    mat_mul88_inline(base2_hp8, eff2_hm8, M2);

    // --- Pass 1: build end-effector dq for each arm (no prefixes stored) -----
    T x1[8], x2[8];
    dq_identity(x1);
    dq_identity(x2);

    // chain 1
    for (int i = 0; i < ith1; ++i) {
        T a[8];
        if (dh1_type == 1) { mdh2dq_inline(dh1, theta1_local, i, a); }
        else               { dh2dq_inline (dh1, theta1_local, i, a); }
        T tmp[8];
        dq_mult_inline(x1, a, tmp);
        dq_copy8(tmp, x1);
    }

    // chain 2
    for (int i = 0; i < ith2; ++i) {
        T a[8];
        if (dh2_type == 1) { mdh2dq_inline(dh2, theta2_local, i, a); }
        else               { dh2dq_inline (dh2, theta2_local, i, a); }
        T tmp[8];
        dq_mult_inline(x2, a, tmp);
        dq_copy8(tmp, x2);
    }

    // --- Build left-projection vectors v1/v2 ---------------------------------
    T x2_conj[8]; conj_inline(x2, x2_conj);

    T hp8_x2c[64]; hamiplus8_inline(x2_conj, hp8_x2c);
    T hm8_x1[64];  haminus8_inline(x1,      hm8_x1);

    T desire_hm8[64]; haminus8_inline(desire_rel_pose, desire_hm8);

    // v = desire_hm8^T * (D8 * dq_tmp)
    T sdq[8]; dq_apply_D8(dq_tmp, sdq);
    T v[8];  dq_mul8T_vec(desire_hm8, sdq, v);

    // v1 = (M1^T) * (hp8_x2c^T * v)
    T t1[8]; dq_mul8T_vec(hp8_x2c, v, t1);
    T v1[8]; dq_mul8T_vec(M1,      t1, v1);

    // v2 = C8 * (hm8_x1^T * v)
    T t2[8]; dq_mul8T_vec(hm8_x1,  v, t2);
    T v2[8]; dq_apply_C8(t2, v2);

    // --- Pass 2: iterate columns with cumulative prefixes --------------------
    // chain 1
    {
        T prefix[8]; dq_identity(prefix);
        for (int i = 0; i < ith1; ++i) {
            // Build a_i (BEFORE updating prefix, since Ad uses prefix up to i-1)
            T a[8];
            if (dh1_type == 1) { mdh2dq_inline(dh1, theta1_local, i, a); }
            else               { dh2dq_inline (dh1, theta1_local, i, a); }

            // j1_i
            T w[8]; get_w_inline(dh1, i, dh1_type, w);
            T z[8]; Ad_inline(prefix, w, z);
#pragma unroll
            for (int k=0;k<8;++k) z[k] = T(0.5) * z[k];

            T j[8]; dq_mult_inline(z, x1, j);
            joint_vel_dir[i] = dq_dot8(v1, j);

            // prefix_{i} -> prefix_{i+1}
            T newp[8]; dq_mult_inline(prefix, a, newp);
            dq_copy8(newp, prefix);
        }
    }

    // chain 2
    {
        T prefix[8]; dq_identity(prefix);
        for (int i = 0; i < ith2; ++i) {
            T a[8];
            if (dh2_type == 1) { mdh2dq_inline(dh2, theta2_local, i, a); }
            else               { dh2dq_inline (dh2, theta2_local, i, a); }

            T w[8]; get_w_inline(dh2, i, dh2_type, w);
            T z[8]; Ad_inline(prefix, w, z);
#pragma unroll
            for (int k=0;k<8;++k) z[k] = T(0.5) * z[k];

            T j[8]; dq_mult_inline(z, x2, j);
            joint_vel_dir[ith1 + i] = dq_dot8(v2, j);

            T newp[8]; dq_mult_inline(prefix, a, newp);
            dq_copy8(newp, prefix);
        }
    }
}

#endif // DQ_FAST_PROJECTION_CUH_
