import torch


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate.

    Expects (..., 4) with ordering (w, x, y, z).
    """
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


def quat_right_matrix(q: torch.Tensor) -> torch.Tensor:
    """Right-multiplication matrix R(q) such that a ⊗ q = R(q) a.

    Expects (..., 4) with ordering (w, x, y, z).
    Returns (..., 4, 4).
    """
    w, x, y, z = q.unbind(dim=-1)
    row0 = torch.stack((w, -x, -y, -z), dim=-1)
    row1 = torch.stack((x, w, z, -y), dim=-1)
    row2 = torch.stack((y, -z, w, x), dim=-1)
    row3 = torch.stack((z, y, -x, w), dim=-1)
    return torch.stack((row0, row1, row2, row3), dim=-2)


def dq_pose_to_twist_map(dq_pose_vec8: torch.Tensor) -> torch.Tensor:
    """Build linear map A(q) from dual-quaternion time-derivative to 6D twist.

    Uses the body-twist relation:
      xi_hat = 2 * qdot_hat ⊗ q_hat*
    with q_hat = q_r + ε q_d.

    Assumes vec8 ordering: [q_r(wxyz), q_d(wxyz)].

    Args:
      dq_pose_vec8: (..., 8)

    Returns:
      A: (..., 6, 8) such that twist6 = A @ qdot8
           twist6 = [ωx, ωy, ωz, vx, vy, vz]
    """
    if dq_pose_vec8.shape[-1] != 8:
        raise ValueError(f"dq_pose_vec8 must have last dim 8, got {dq_pose_vec8.shape}")

    qr = dq_pose_vec8[..., 0:4]
    qd = dq_pose_vec8[..., 4:8]
    qr_c = quat_conj(qr)
    qd_c = quat_conj(qd)

    R_qr_c = quat_right_matrix(qr_c)  # (..., 4, 4)
    R_qd_c = quat_right_matrix(qd_c)  # (..., 4, 4)

    # V selects vector part (x, y, z) from quaternion (w, x, y, z)
    V = dq_pose_vec8.new_zeros((3, 4))
    V[0, 1] = 1.0
    V[1, 2] = 1.0
    V[2, 3] = 1.0

    # Broadcast V to batch shape
    # omega = 2 * V * (R(qr*) qdot_r)
    A_omega_r = 2.0 * (V @ R_qr_c)  # (..., 3, 4)
    A_omega_d = dq_pose_vec8.new_zeros(A_omega_r.shape)

    # v = 2 * V * (R(qr*) qdot_d + R(qd*) qdot_r)
    A_v_r = 2.0 * (V @ R_qd_c)  # (..., 3, 4)
    A_v_d = 2.0 * (V @ R_qr_c)  # (..., 3, 4)

    top = torch.cat((A_omega_r, A_omega_d), dim=-1)  # (..., 3, 8)
    bot = torch.cat((A_v_r, A_v_d), dim=-1)  # (..., 3, 8)
    return torch.cat((top, bot), dim=-2)  # (..., 6, 8)


def dq_jacobian_to_twist_jacobian(dq_pose_vec8: torch.Tensor, dq_jacobian: torch.Tensor) -> torch.Tensor:
    """Convert DQ pose Jacobian (8xN) into a 6xN twist Jacobian.

    Args:
      dq_pose_vec8: (..., 8)
      dq_jacobian: (..., 8, N)

    Returns:
      J_twist: (..., 6, N)
    """
    if dq_jacobian.shape[-2] != 8:
        raise ValueError(f"dq_jacobian must have shape (..., 8, N), got {dq_jacobian.shape}")
    A = dq_pose_to_twist_map(dq_pose_vec8)  # (..., 6, 8)
    return A @ dq_jacobian


def manipulability_svd(jacobian: torch.Tensor, k: int = 6, eps: float = 1e-12) -> torch.Tensor:
    """Pseudo-determinant manipulability via singular values.

    Works for rank-deficient Jacobians like 8x12 DQ Jacobians (rank <= 6).

    Args:
      jacobian: (..., M, N)
      k: number of leading singular values to multiply (default 6)
      eps: clamp lower bound for numerical stability

    Returns:
      manip: (..., 1)
    """
    s = torch.linalg.svdvals(jacobian)  # (..., min(M,N)) in descending order
    k_eff = min(k, s.shape[-1])
    s_k = torch.clamp(s[..., :k_eff], min=eps)
    manip = torch.prod(s_k, dim=-1, keepdim=True)
    return manip


def manipulability_pdet(jacobian: torch.Tensor, k: int | None = None, eps: float = 1e-12) -> torch.Tensor:
    """Pseudo-determinant manipulability via singular values product.

    Args:
      jacobian: (..., M, N)
      k: number of leading singular values to multiply. If None, uses min(M, N).
      eps: clamp lower bound for numerical stability

    Returns:
      manip: (..., 1)
    """
    if jacobian.ndim < 2:
        raise ValueError(f"jacobian must be at least 2D (..., M, N), got {jacobian.shape}")

    s = torch.linalg.svdvals(jacobian)  # (..., min(M,N))
    if k is None:
        k_eff = s.shape[-1]
    else:
        k_eff = min(int(k), s.shape[-1])
    s_k = torch.clamp(s[..., :k_eff], min=eps)
    return torch.prod(s_k, dim=-1, keepdim=True)


def manipulability_cost_from_jacobian_pdet(
    jacobian: torch.Tensor,
    weight: float = 1.0,
    k: int | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Convenience: cost = weight / pdet_manipulability."""
    manip = manipulability_pdet(jacobian, k=k, eps=eps)
    return weight / torch.clamp(manip, min=eps)


def manipulability_cost_from_jacobian(jacobian: torch.Tensor, weight: float = 1.0, k: int = 6, eps: float = 1e-12) -> torch.Tensor:
    """Convenience: cost = weight / manipulability."""
    manip = manipulability_svd(jacobian, k=k, eps=eps)
    return weight / torch.clamp(manip, min=eps)


def manipulability_yoshikawa_from_twist_jacobian(j_twist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Yoshikawa manipulability for a 6xN twist Jacobian.

    Computes sqrt(det(J J^T)) with symmetric stabilization.

    Args:
      j_twist: (..., 6, N)
      eps: diagonal regularization for numerical stability

    Returns:
      manip: (..., 1)
    """
    if j_twist.shape[-2] != 6:
        raise ValueError(f"j_twist must have shape (..., 6, N), got {j_twist.shape}")

    gram = j_twist @ j_twist.transpose(-2, -1)  # (..., 6, 6)
    sym = 0.5 * (gram + gram.transpose(-2, -1))
    eye = torch.eye(6, device=j_twist.device, dtype=j_twist.dtype)
    sym = sym + eps * eye
    det64 = torch.linalg.det(sym.to(torch.float64))
    det64 = torch.clamp(det64, min=float(eps))
    manip64 = torch.sqrt(det64)
    return manip64.to(j_twist.dtype).unsqueeze(-1)


def manipulability_yoshikawa_from_jacobian(jacobian: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Yoshikawa manipulability for a kxN Jacobian.

    Computes sqrt(det(J J^T)) with symmetric stabilization.

    Args:
      jacobian: (..., k, N)
      eps: diagonal regularization for numerical stability

    Returns:
      manip: (..., 1)
    """
    if jacobian.ndim < 2:
        raise ValueError(f"jacobian must be at least 2D (..., k, N), got {jacobian.shape}")
    k = jacobian.shape[-2]
    gram = jacobian @ jacobian.transpose(-2, -1)  # (..., k, k)
    sym = 0.5 * (gram + gram.transpose(-2, -1))
    eye = torch.eye(k, device=jacobian.device, dtype=jacobian.dtype)
    sym = sym + eps * eye
    det64 = torch.linalg.det(sym.to(torch.float64))
    det64 = torch.clamp(det64, min=float(eps))
    manip64 = torch.sqrt(det64)
    return manip64.to(jacobian.dtype).unsqueeze(-1)


def manipulability_from_gram_eig_scaled(
    jacobian: torch.Tensor,
    last_k: int = 3,
    div: float = 10.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Manipulability from eigenvalues of Gram matrix with scaled tail.

    Let G = J J^T (k x k). Compute eigenvalues λ (ascending). Replace the
    largest `last_k` eigenvalues by λ/div, then compute:

      w = sqrt(prod( clamp(λ_scaled, eps) ))

    Args:
      jacobian: (..., k, N)
      last_k: how many largest eigenvalues to scale (default 3)
      div: division factor for those eigenvalues (default 10)
      eps: clamp for numerical stability

    Returns:
      manip: (..., 1)
    """
    if jacobian.ndim < 2:
        raise ValueError(f"jacobian must be at least 2D (..., k, N), got {jacobian.shape}")
    k_dim = jacobian.shape[-2]
    if k_dim <= 0:
        raise ValueError(f"invalid jacobian shape {jacobian.shape}")
    last_k = max(int(last_k), 0)

    gram = jacobian @ jacobian.transpose(-2, -1)
    gram = 0.5 * (gram + gram.transpose(-2, -1))
    eig64 = torch.linalg.eigvalsh(gram.to(torch.float64))  # (..., k)

    eig_scaled = eig64
    if last_k > 0 and float(div) != 1.0 and k_dim >= last_k:
        eig_scaled = eig64.clone()
        eig_scaled[..., -last_k:] = eig_scaled[..., -last_k:] / float(div)

    eig_scaled = torch.clamp(eig_scaled, min=float(eps))
    prod64 = torch.prod(eig_scaled, dim=-1, keepdim=True)  # (..., 1)
    manip64 = torch.sqrt(prod64)
    return manip64.to(jacobian.dtype)


def manipulability_cost_from_gram_eig_scaled(
  jacobian: torch.Tensor,
  weight: float = 1.0,
  last_k: int = 3,
  div: float = 10.0,
  eps: float = 1e-12,
) -> torch.Tensor:
  manip = manipulability_from_gram_eig_scaled(jacobian, last_k=last_k, div=div, eps=eps)
  return weight / torch.clamp(manip, min=eps)


def manipulability_cost_from_twist_jacobian(j_twist: torch.Tensor, weight: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """Convenience: cost = weight / Yoshikawa manipulability on 6xN twist Jacobian."""
    manip = manipulability_yoshikawa_from_twist_jacobian(j_twist, eps=eps)
    return weight / torch.clamp(manip, min=eps)


def manipulability_cost_from_twist_jacobian_rows(
  j_twist: torch.Tensor,
  weight: float = 1.0,
  rows: str = "all6",
  eps: float = 1e-12,
) -> torch.Tensor:
  """Convenience: cost from selected rows of a 6xN twist Jacobian.

  rows options:
    - "all6": use all 6 rows (standard twist manipulability)
    - "first3": use rows [0:3] only
    - "last3": use rows [3:6] only
  """
  if rows in ("all6", "all", "6"):
    return manipulability_cost_from_twist_jacobian(j_twist, weight=weight, eps=eps)

  if j_twist.shape[-2] != 6:
    raise ValueError(f"j_twist must have shape (..., 6, N), got {j_twist.shape}")

  if rows == "first3":
    j_part = j_twist[..., :3, :]
  elif rows == "last3":
    j_part = j_twist[..., 3:6, :]
  else:
    raise ValueError(f"rows must be one of 'all6', 'first3', 'last3' (got {rows!r})")

  manip = manipulability_yoshikawa_from_jacobian(j_part, eps=eps)
  return weight / torch.clamp(manip, min=eps)


def manipulability_from_twist_jacobian_rows_pdet(
  j_twist: torch.Tensor,
  rows: str = "all6",
  k: int | None = None,
  eps: float = 1e-12,
) -> torch.Tensor:
  """Return pdet manipulability from selected rows of a 6xN twist Jacobian."""
  if rows in ("all6", "all", "6"):
    j_part = j_twist
  else:
    if j_twist.shape[-2] != 6:
      raise ValueError(f"j_twist must have shape (..., 6, N), got {j_twist.shape}")
    if rows == "first3":
      j_part = j_twist[..., :3, :]
    elif rows == "last3":
      j_part = j_twist[..., 3:6, :]
    else:
      raise ValueError(f"rows must be one of 'all6', 'first3', 'last3' (got {rows!r})")

  if k is None:
    k = j_part.shape[-2]
  return manipulability_pdet(j_part, k=int(k), eps=eps)


def manipulability_from_twist_jacobian_rows_gram_eig_scaled(
  j_twist: torch.Tensor,
  rows: str = "all6",
  last_k: int = 3,
  div: float = 10.0,
  eps: float = 1e-12,
) -> torch.Tensor:
  """Return gram-eigen scaled manipulability from selected rows of 6xN twist Jacobian."""
  if rows in ("all6", "all", "6"):
    j_part = j_twist
  else:
    if j_twist.shape[-2] != 6:
      raise ValueError(f"j_twist must have shape (..., 6, N), got {j_twist.shape}")
    if rows == "first3":
      j_part = j_twist[..., :3, :]
    elif rows == "last3":
      j_part = j_twist[..., 3:6, :]
    else:
      raise ValueError(f"rows must be one of 'all6', 'first3', 'last3' (got {rows!r})")

  # For 3D, scaling last 3 means scaling all eigenvalues; allow last_k to be clamped.
  k_dim = j_part.shape[-2]
  lk = min(max(int(last_k), 0), int(k_dim))
  return manipulability_from_gram_eig_scaled(j_part, last_k=lk, div=div, eps=eps)


def manipulability_cost_from_twist_jacobian_rows_gram_eig_scaled(
  j_twist: torch.Tensor,
  weight: float = 1.0,
  rows: str = "all6",
  last_k: int = 3,
  div: float = 10.0,
  eps: float = 1e-12,
) -> torch.Tensor:
  manip = manipulability_from_twist_jacobian_rows_gram_eig_scaled(
    j_twist,
    rows=rows,
    last_k=last_k,
    div=div,
    eps=eps,
  )
  return weight / torch.clamp(manip, min=eps)


def manipulability_cost_from_twist_jacobian_rows_pdet(
  j_twist: torch.Tensor,
  weight: float = 1.0,
  rows: str = "all6",
  k: int | None = None,
  eps: float = 1e-12,
) -> torch.Tensor:
  """Convenience: cost = weight / pdet manipulability on selected twist Jacobian rows."""
  manip = manipulability_from_twist_jacobian_rows_pdet(j_twist, rows=rows, k=k, eps=eps)
  return weight / torch.clamp(manip, min=eps)


def manipulability_from_twist_jacobian_rows(
  j_twist: torch.Tensor,
  rows: str = "all6",
  eps: float = 1e-12,
) -> torch.Tensor:
  """Return Yoshikawa manipulability from selected rows of a 6xN twist Jacobian.

  rows options:
    - "all6": use all 6 rows
    - "first3": use rows [0:3]
    - "last3": use rows [3:6]
  """
  if rows in ("all6", "all", "6"):
    return manipulability_yoshikawa_from_twist_jacobian(j_twist, eps=eps)

  if j_twist.shape[-2] != 6:
    raise ValueError(f"j_twist must have shape (..., 6, N), got {j_twist.shape}")

  if rows == "first3":
    j_part = j_twist[..., :3, :]
  elif rows == "last3":
    j_part = j_twist[..., 3:6, :]
  else:
    raise ValueError(f"rows must be one of 'all6', 'first3', 'last3' (got {rows!r})")

  return manipulability_yoshikawa_from_jacobian(j_part, eps=eps)
