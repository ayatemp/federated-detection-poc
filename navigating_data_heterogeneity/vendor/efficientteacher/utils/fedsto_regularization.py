import math

import torch


def _parameter_in_scope(name, scope):
    lowered = name.lower()
    is_bn = ".bn." in lowered or "batchnorm" in lowered
    if scope == "all":
        return True
    if scope == "non_backbone":
        return "backbone" not in lowered
    if scope == "neck_head":
        return ("neck" in lowered) or ("head" in lowered)
    if scope == "bn":
        return is_bn
    if scope == "bn_moe_head":
        return is_bn or ("head.router" in lowered) or ("head.expert_m" in lowered)
    if scope == "moe_head":
        return ("head.router" in lowered) or ("head.expert_m" in lowered)
    if scope == "backbone_moe_head":
        return ("backbone" in lowered) or ("head.router" in lowered) or ("head.expert_m" in lowered)
    return False


def _spectral_norm_power_iteration(matrix, iterations=3):
    size = matrix.shape[0]
    if size == 0:
        return matrix.new_zeros(())
    with torch.no_grad():
        vector = torch.arange(1, size + 1, device=matrix.device, dtype=matrix.dtype).unsqueeze(1)
        vector = torch.nn.functional.normalize(vector, dim=0, eps=1e-12)
        detached = matrix.detach()
        for _ in range(iterations):
            vector = detached.matmul(vector)
            vector = torch.nn.functional.normalize(vector, dim=0, eps=1e-12)
    return matrix.matmul(vector).norm()


def spectral_orthogonal_regularization(model, weight=0.0, scope="non_backbone"):
    if weight <= 0:
        return None

    penalty = None
    count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.ndim < 2 or not _parameter_in_scope(name, scope):
            continue
        param_device = param.device
        matrix = param.float().reshape(param.shape[0], -1).contiguous()
        if matrix.numel() == 0:
            continue
        # Avoid intermittent cuBLAS failures from the full Gram product during Phase 2.
        if matrix.is_cuda:
            matrix = matrix.cpu()
        gram = torch.mm(matrix, matrix.transpose(0, 1).contiguous())
        gram = gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        layer_penalty = _spectral_norm_power_iteration(gram)
        if layer_penalty.device != param_device:
            layer_penalty = layer_penalty.to(param_device)
        penalty = layer_penalty if penalty is None else penalty + layer_penalty
        count += 1

    if penalty is None:
        first_param = next(model.parameters())
        return first_param.new_zeros(())
    return penalty.mul(float(weight) / max(count, 1))


def apply_fedsto_train_scope(model, scope="all"):
    if scope == "all":
        for param in model.parameters():
            param.requires_grad = True
        return

    for name, param in model.named_parameters():
        lowered = name.lower()
        is_bn = ".bn." in lowered or "batchnorm" in lowered
        if scope == "backbone":
            param.requires_grad = "backbone" in lowered
        elif scope == "non_backbone":
            param.requires_grad = "backbone" not in lowered
        elif scope == "neck_head":
            param.requires_grad = ("neck" in lowered) or ("head" in lowered)
        elif scope == "bn":
            param.requires_grad = is_bn
        elif scope == "bn_moe_head":
            param.requires_grad = is_bn or ("head.router" in lowered) or ("head.expert_m" in lowered)
        elif scope == "moe_head":
            param.requires_grad = ("head.router" in lowered) or ("head.expert_m" in lowered)
        elif scope == "backbone_moe_head":
            param.requires_grad = (
                "backbone" in lowered
                or ("head.router" in lowered)
                or ("head.expert_m" in lowered)
            )
        else:
            raise ValueError(f"Unsupported FedSTO.train_scope: {scope}")


def _base_model(model):
    return model.module if hasattr(model, "module") else model


def _class_rows(conv, na, no, nc):
    weight = conv.weight.float().reshape(conv.out_channels, -1)
    rows = []
    for anchor_idx in range(na):
        start = anchor_idx * no + 5
        rows.append(weight[start:start + nc])
    return torch.cat(rows, dim=0)


def _identity_like_gram(gram):
    return torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)


def class_skew_head_regularization(
    model,
    orthogonal_weight=0.0,
    srip_weight=0.0,
    residual_weight=0.0,
):
    if orthogonal_weight <= 0 and srip_weight <= 0 and residual_weight <= 0:
        return None

    base = _base_model(model)
    head = getattr(base, "head", None)
    if head is None or not getattr(head, "class_skew_enabled", False) or not hasattr(head, "residual_m"):
        return None

    total = None
    count = 0
    for shared_conv, residual_conv in zip(head.m, head.residual_m):
        shared = _class_rows(shared_conv, head.na, head.no, head.nc)
        residual = residual_conv.weight.float().reshape(residual_conv.out_channels, -1)
        layer_loss = shared.new_zeros(())

        if orthogonal_weight > 0:
            overlap = shared @ residual.transpose(0, 1)
            layer_loss = layer_loss + float(orthogonal_weight) * overlap.square().mean()

        if srip_weight > 0:
            gram = shared @ shared.transpose(0, 1)
            srip = torch.linalg.matrix_norm(gram - _identity_like_gram(gram), ord=2)
            layer_loss = layer_loss + float(srip_weight) * srip

        if residual_weight > 0:
            layer_loss = layer_loss + float(residual_weight) * residual.square().mean()

        total = layer_loss if total is None else total + layer_loss
        count += 1

    if total is None:
        first_param = next(base.parameters())
        return first_param.new_zeros(())
    return total / max(count, 1)


def latent_moe_router_regularization(
    model,
    balance_weight=0.0,
    entropy_weight=0.0,
    specialization_weight=0.0,
    specialization_target=-1,
):
    if balance_weight <= 0 and entropy_weight <= 0 and specialization_weight <= 0:
        return None

    base = _base_model(model)
    head = getattr(base, "head", None)
    probs_list = getattr(head, "last_router_probs", None)
    if head is None or not probs_list:
        return None

    total = None
    count = 0
    for probs in probs_list:
        probs = probs.float()
        if probs.ndim != 4 or probs.shape[1] <= 1:
            continue
        num_experts = probs.shape[1]
        importance = probs.mean(dim=(0, 2, 3))
        layer_loss = probs.new_zeros(())

        if balance_weight > 0:
            target = importance.new_full((num_experts,), 1.0 / num_experts)
            layer_loss = layer_loss + float(balance_weight) * num_experts * (importance - target).square().sum()

        if entropy_weight > 0:
            safe_probs = probs.clamp_min(1e-8)
            entropy = -(safe_probs * safe_probs.log()).sum(dim=1).mean()
            entropy = entropy / math.log(max(num_experts, 2))
            layer_loss = layer_loss + float(entropy_weight) * (1.0 - entropy)

        target_idx = int(specialization_target)
        if specialization_weight > 0 and 0 <= target_idx < num_experts:
            target_probs = probs[:, target_idx, :, :].clamp_min(1e-8)
            layer_loss = layer_loss + float(specialization_weight) * (-target_probs.log().mean())

        total = layer_loss if total is None else total + layer_loss
        count += 1

    if total is None:
        first_param = next(base.parameters())
        return first_param.new_zeros(())
    return total / max(count, 1)


def clear_latent_moe_router_cache(model) -> None:
    base = _base_model(model)
    head = getattr(base, "head", None)
    if head is not None and hasattr(head, "last_router_probs"):
        head.last_router_probs = []
