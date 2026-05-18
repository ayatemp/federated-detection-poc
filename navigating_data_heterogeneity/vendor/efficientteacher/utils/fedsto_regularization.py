import math

import torch


def _parameter_in_scope(name, scope):
    lowered = name.lower()
    is_adapter = "backbone.adapter_moe" in lowered or "neck.adapter_moe" in lowered
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
    if scope == "backbone_adapter_moe":
        return is_adapter
    if scope == "backbone_adapter_moe_head_moe":
        return is_adapter or ("head.router" in lowered) or ("head.expert_m" in lowered)
    if scope == "backbone_adapter_moe_head":
        return is_adapter or ("head" in lowered)
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
        is_adapter = "backbone.adapter_moe" in lowered or "neck.adapter_moe" in lowered
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
        elif scope == "backbone_adapter_moe":
            param.requires_grad = is_adapter
        elif scope == "backbone_adapter_moe_head_moe":
            param.requires_grad = is_adapter or ("head.router" in lowered) or ("head.expert_m" in lowered)
        elif scope == "backbone_adapter_moe_head":
            param.requires_grad = is_adapter or ("head" in lowered)
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
    sample_entropy_weight=0.0,
    specialization_weight=0.0,
    specialization_target=-1,
):
    if balance_weight <= 0 and entropy_weight <= 0 and sample_entropy_weight <= 0 and specialization_weight <= 0:
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

        safe_probs = probs.clamp_min(1e-8)
        if entropy_weight > 0 or sample_entropy_weight > 0:
            entropy = -(safe_probs * safe_probs.log()).sum(dim=1).mean()
            entropy = entropy / math.log(max(num_experts, 2))

        if entropy_weight > 0:
            layer_loss = layer_loss + float(entropy_weight) * (1.0 - entropy)

        if sample_entropy_weight > 0:
            layer_loss = layer_loss + float(sample_entropy_weight) * entropy

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


def _cached_tensors_from_module(module, attr_name):
    tensors = []
    for submodule in module.modules():
        cached = getattr(submodule, "__dict__", {}).get(attr_name, None)
        if cached:
            tensors.extend(cached)
    return tensors


def _adapter_expert_diversity_loss(backbone, weight):
    if weight <= 0:
        return None

    total = None
    count = 0
    for module in backbone.modules():
        experts = getattr(module, "experts", None)
        if experts is None or len(experts) <= 1:
            continue
        vectors = []
        for expert in experts:
            reduce = getattr(expert, "reduce", None)
            if reduce is None:
                continue
            vectors.append(reduce.weight.float().flatten())
        if len(vectors) <= 1:
            continue
        matrix = torch.stack(vectors, dim=0)
        matrix = torch.nn.functional.normalize(matrix, dim=1, eps=1e-8)
        gram = matrix @ matrix.transpose(0, 1)
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        layer_loss = (gram - eye).square().mean()
        total = layer_loss if total is None else total + layer_loss
        count += 1

    if total is None:
        return None
    return total.mul(float(weight) / max(count, 1))


def _adapter_router_regularization_for_module(
    module,
    balance_weight=0.0,
    entropy_weight=0.0,
    sample_entropy_weight=0.0,
    z_loss_weight=0.0,
    diversity_weight=0.0,
):
    if (
        balance_weight <= 0
        and entropy_weight <= 0
        and sample_entropy_weight <= 0
        and z_loss_weight <= 0
        and diversity_weight <= 0
    ):
        return None

    probs_list = _cached_tensors_from_module(module, "last_router_probs")
    logits_list = _cached_tensors_from_module(module, "last_router_logits")

    total = None
    count = 0
    for probs in probs_list:
        probs = probs.float()
        if probs.ndim == 2:
            importance = probs.mean(dim=0)
            entropy_dims = 1
        elif probs.ndim == 4:
            importance = probs.mean(dim=(0, 2, 3))
            entropy_dims = 1
        else:
            continue
        num_experts = probs.shape[1]
        layer_loss = probs.new_zeros(())

        if balance_weight > 0:
            target = importance.new_full((num_experts,), 1.0 / num_experts)
            layer_loss = layer_loss + float(balance_weight) * num_experts * (importance - target).square().sum()

        safe_probs = probs.clamp_min(1e-8)
        if entropy_weight > 0 or sample_entropy_weight > 0:
            entropy = -(safe_probs * safe_probs.log()).sum(dim=entropy_dims).mean()
            entropy = entropy / math.log(max(num_experts, 2))

        if entropy_weight > 0:
            layer_loss = layer_loss + float(entropy_weight) * (1.0 - entropy)

        if sample_entropy_weight > 0:
            layer_loss = layer_loss + float(sample_entropy_weight) * entropy

        total = layer_loss if total is None else total + layer_loss
        count += 1

    if z_loss_weight > 0:
        for logits in logits_list:
            if logits.ndim != 2:
                continue
            z_loss = torch.logsumexp(logits.float(), dim=1).square().mean()
            total = z_loss.mul(float(z_loss_weight)) if total is None else total + z_loss.mul(float(z_loss_weight))
            count += 1

    diversity_loss = _adapter_expert_diversity_loss(module, diversity_weight)
    if diversity_loss is not None:
        total = diversity_loss if total is None else total + diversity_loss
        count += 1

    if total is None:
        first_param = next(module.parameters())
        return first_param.new_zeros(())
    return total / max(count, 1)


def backbone_moe_router_regularization(
    model,
    balance_weight=0.0,
    entropy_weight=0.0,
    sample_entropy_weight=0.0,
    z_loss_weight=0.0,
    diversity_weight=0.0,
    neck_balance_weight=None,
    neck_entropy_weight=None,
    neck_sample_entropy_weight=None,
    neck_z_loss_weight=None,
    neck_diversity_weight=None,
):
    base = _base_model(model)
    backbone = getattr(base, "backbone", None)
    neck = getattr(base, "neck", None)
    total = None
    count = 0

    if backbone is not None:
        loss = _adapter_router_regularization_for_module(
            backbone,
            balance_weight,
            entropy_weight,
            sample_entropy_weight,
            z_loss_weight,
            diversity_weight,
        )
        if loss is not None:
            total = loss if total is None else total + loss
            count += 1

    if neck is not None:
        loss = _adapter_router_regularization_for_module(
            neck,
            balance_weight if neck_balance_weight is None else neck_balance_weight,
            entropy_weight if neck_entropy_weight is None else neck_entropy_weight,
            sample_entropy_weight if neck_sample_entropy_weight is None else neck_sample_entropy_weight,
            z_loss_weight if neck_z_loss_weight is None else neck_z_loss_weight,
            diversity_weight if neck_diversity_weight is None else neck_diversity_weight,
        )
        if loss is not None:
            total = loss if total is None else total + loss
            count += 1

    if total is None:
        first_param = next(base.parameters())
        return first_param.new_zeros(())
    return total / max(count, 1)


def clear_latent_moe_router_cache(model) -> None:
    base = _base_model(model)
    for module in (getattr(base, "head", None), getattr(base, "backbone", None), getattr(base, "neck", None)):
        if module is None:
            continue
        for submodule in module.modules():
            for attr_name in ("last_router_probs", "last_router_logits", "last_router_hard_probs"):
                if attr_name in getattr(submodule, "__dict__", {}):
                    setattr(submodule, attr_name, [])
                elif isinstance(getattr(type(submodule), attr_name, None), property):
                    setattr(submodule, attr_name, [])
