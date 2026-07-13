import torch

from como.geometry.camera import projection, transform_project
from como.geometry.lie_algebra import se3_exp, skew_symmetric
from como.odom.frontend.photo_utils import img_interp
import como.odom.backend.robust_loss as robust


def _scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


# Coarse-to-fine where inputs are lists except Tji_init
def photo_tracking_pyr(
    Tji_init,
    aff_init,
    vals_i,
    Pi,
    dI_dT,
    masks,
    intrinsics,
    img_j,
    photo_sigma,
    term_criteria,
    return_debug=False,
):
    Tji = Tji_init.clone()
    aff = aff_init.clone()
    num_levels = len(vals_i)
    debug_levels = []

    for l in range(num_levels):
        mask_l = masks[l]
        vals_l = vals_i[l][None, mask_l, :]
        P_l = Pi[l][None, mask_l, :]
        dI_dT_l = dI_dT[l][None, mask_l, :, :]

        Tji, aff, debug_level = photo_level_tracking(
            Tji,
            aff,
            vals_l,
            P_l,
            dI_dT_l,
            img_j[l],
            intrinsics[l],
            photo_sigma,
            term_criteria,
        )

        if return_debug:
            debug_level["level"] = l
            debug_levels.append(debug_level)

    if return_debug:
        return Tji, aff, debug_levels
    return Tji, aff


# IC precalculate Jacobians at theta=0
def precalc_jacobians(dI_dw, P, vals, intrinsics):
    c = vals.shape[2]
    device = dI_dw.device
    dtype = dI_dw.dtype

    b, n, _ = P.shape
    dPi_dT = torch.empty((b, n, 3, 6), device=device, dtype=dtype)
    dPi_dT[:, :, :, 3:] = (
        torch.eye(3, device=device, dtype=dtype)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(b, n, 1, 1)
    )
    dPi_dT[:, :, :, :3] = -skew_symmetric(P)

    _, dpi_dPi = projection(intrinsics, P)
    dpi_dT = torch.matmul(dpi_dPi, dPi_dT)
    dI_dT = torch.matmul(dI_dw, dpi_dT)

    dI_dp = torch.cat(
        (
            dI_dT,
            vals.unsqueeze(-1),
            torch.ones((dI_dT.shape[0], n, c, 1), device=device, dtype=dtype),
        ),
        dim=-1,
    )

    return dI_dp


def robustify_photo(r, dIt_dT, invalid_mask, photo_sigma):
    info_sqrt = 1.0 / photo_sigma
    whitened_r = r * info_sqrt
    weight = robust.huber(whitened_r)
    weight[invalid_mask[...], :] = 0.0

    total_err = torch.sum(weight * torch.square(whitened_r))
    num_valid = invalid_mask.shape[-1] - torch.count_nonzero(invalid_mask, dim=-1)
    mean_sq_err = total_err / num_valid

    J_W = weight[:, :, :, None] * dIt_dT
    grad = torch.sum(J_W * r[..., None], dim=(1, 2))
    H = torch.einsum("bnck,bncl->bkl", J_W, dIt_dT)

    grad_norm = torch.linalg.norm(grad)

    return H, grad, total_err, mean_sq_err, grad_norm


def solve_delta(H, grad):
    L, info = torch.linalg.cholesky_ex(H, upper=False, check_errors=False)
    delta = torch.cholesky_solve(grad[..., None], L, upper=False)
    return delta, info


# TODO: Batch
def update_pose_ic(T, aff, delta):
    delta_T = delta[:, :6, 0]
    T_new = torch.matmul(T, se3_exp(-delta_T))

    delta_a = delta[:, 6, 0]
    delta_b = delta[:, 7, 0]
    aff_new = torch.empty_like(aff)

    aff_new[:, 0] = aff[:, 0] - delta_a
    aff_new[:, 1] = aff[:, 1] - delta_b

    return T_new, aff_new


def tracking_iter(Tji, Pi, intrinsics, img_j, aff, vals_i, dI_dT, photo_sigma, A_norm):
    pj, depth_j = transform_project(intrinsics, Tji, Pi)

    vals_target, valid_mask = img_interp(img_j, pj, A_norm)
    valid_mask = torch.logical_and(valid_mask, depth_j[..., 0] > 0)
    invalid_mask = torch.logical_not(valid_mask)

    tmp = torch.exp(-aff[:, None, 0]) * vals_target
    dI_dT[..., 6] = torch.permute(-tmp, (0, 2, 1))
    vals_target = tmp + aff[:, None, 1]

    vals_ref = torch.permute(vals_i, (0, 2, 1))
    r = vals_target - vals_ref
    r = torch.permute(r, (0, 2, 1))

    valid_count = torch.count_nonzero(valid_mask)
    total_count = valid_mask.numel()

    if valid_count > 0:
        r_abs = torch.abs(r[valid_mask])
        med_r = torch.median(r_abs)
        sigma_r = 1.4826 * med_r
        residual_abs_mean = torch.mean(r_abs)
        residual_abs_max = torch.max(r_abs)
    else:
        med_r = torch.tensor(float("nan"), device=r.device, dtype=r.dtype)
        sigma_r = torch.tensor(float("nan"), device=r.device, dtype=r.dtype)
        residual_abs_mean = torch.tensor(float("nan"), device=r.device, dtype=r.dtype)
        residual_abs_max = torch.tensor(float("nan"), device=r.device, dtype=r.dtype)

    H, grad, total_err, mean_sq_err, grad_norm = robustify_photo(
        r, dI_dT, invalid_mask, sigma_r
    )

    delta, chol_info = solve_delta(H, grad)
    Tji_new, aff_new = update_pose_ic(Tji, aff, delta)

    H0 = H[0]
    H_diag = torch.diagonal(H0)
    try:
        H_cond = torch.linalg.cond(H0)
    except RuntimeError:
        H_cond = torch.tensor(float("inf"), device=H.device, dtype=H.dtype)

    debug_info = {
        "valid_count": int(valid_count.detach().cpu().item()),
        "total_count": int(total_count),
        "valid_ratio": float(valid_count.detach().cpu().item()) / float(total_count) if total_count > 0 else 0.0,
        "sigma_r": _scalar(sigma_r),
        "residual_abs_median": _scalar(med_r),
        "residual_abs_mean": _scalar(residual_abs_mean),
        "residual_abs_max": _scalar(residual_abs_max),
        "grad_norm": _scalar(grad_norm),
        "delta_norm": _scalar(torch.norm(delta)),
        "h_diag_min": _scalar(torch.min(H_diag)),
        "h_diag_max": _scalar(torch.max(H_diag)),
        "h_cond": _scalar(H_cond),
        "cholesky_ok": bool((chol_info == 0).all().detach().cpu().item()),
        "pose_jac_abs_mean": _scalar(torch.mean(torch.abs(dI_dT[..., :6]))) if dI_dT.numel() > 0 else 0.0,
        "pose_jac_abs_max": _scalar(torch.max(torch.abs(dI_dT[..., :6]))) if dI_dT.numel() > 0 else 0.0,
        "affine_jac_abs_mean": _scalar(torch.mean(torch.abs(dI_dT[..., 6:]))) if dI_dT.numel() > 0 else 0.0,
        "affine_jac_abs_max": _scalar(torch.max(torch.abs(dI_dT[..., 6:]))) if dI_dT.numel() > 0 else 0.0,
        "depth_positive_ratio": _scalar(
            torch.count_nonzero(depth_j[..., 0] > 0) / depth_j[..., 0].numel()
        ),
    }

    return (
        Tji_new,
        aff_new,
        delta,
        mean_sq_err,
        grad_norm,
        pj,
        valid_mask,
        depth_j,
        debug_info,
    )


# Inverse compositional tracking
def photo_level_tracking(
    Tji_init, aff_init, vals_i, Pi, dI_dT, img_j, intrinsics, photo_sigma, term_criteria
):
    Tji = Tji_init.clone()
    aff = aff_init.clone()

    A_norm = 1.0 / torch.as_tensor(
        (img_j.shape[-1], img_j.shape[-2]), device=img_j.device, dtype=img_j.dtype
    )

    iter = 0
    done = False
    mean_sq_err_prev = float("inf")
    debug_iters = []
    stop_reason = "max_iter"

    while not done:
        (
            Tji,
            aff,
            delta,
            mean_sq_err,
            grad_norm,
            p_j,
            valid_reproj_mask,
            depth_j,
            iter_debug,
        ) = (
            tracking_iter(
                Tji, Pi, intrinsics, img_j, aff, vals_i, dI_dT, photo_sigma, A_norm
            )
        )

        iter += 1
        delta_norm = torch.norm(delta)
        abs_decrease = mean_sq_err_prev - mean_sq_err

        # NOTE: Checking for convergence, not if error goes up, so want absolute value!
        rel_decrease = torch.abs(abs_decrease / mean_sq_err_prev)

        iter_debug["iter"] = iter
        iter_debug["mean_sq_err"] = _scalar(mean_sq_err)
        iter_debug["rel_decrease"] = _scalar(rel_decrease)
        debug_iters.append(iter_debug)

        # print("Tracking: ", iter, mean_sq_err.item(), delta_norm.item(), rel_decrease.item(), grad_norm.item())
        if (
            iter >= term_criteria["max_iter"]
            or delta_norm < term_criteria["delta_norm"]
            or rel_decrease < term_criteria["rel_tol"]
            or grad_norm < term_criteria["grad_norm"]
        ):
            done = True
            if iter >= term_criteria["max_iter"]:
                stop_reason = "max_iter"
            elif delta_norm < term_criteria["delta_norm"]:
                stop_reason = "delta_norm"
            elif rel_decrease < term_criteria["rel_tol"]:
                stop_reason = "rel_tol"
            else:
                stop_reason = "grad_norm"

        mean_sq_err_prev = mean_sq_err

    debug_summary = {
        "num_iters": iter,
        "stop_reason": stop_reason,
        "iters": debug_iters,
    }

    return Tji, aff, debug_summary