import torch


def _parameter_in_scope(name, scope):
    lowered = name.lower()
    if scope == "all":
        return True
    if scope == "non_backbone":
        return "backbone" not in lowered
    if scope == "neck_head":
        return ("neck" in lowered) or ("head" in lowered)
    return False


def spectral_orthogonal_regularization(model, weight=0.0, scope="non_backbone"):
    if weight <= 0:
        return None

    penalty = None
    count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.ndim < 2 or not _parameter_in_scope(name, scope):
            continue
        matrix = param.float().reshape(param.shape[0], -1)
        if matrix.numel() == 0:
            continue
        singular_values = torch.linalg.svdvals(matrix)
        layer_penalty = torch.max(torch.abs(singular_values.square() - 1.0))
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
        if scope == "backbone":
            param.requires_grad = "backbone" in lowered
        elif scope == "non_backbone":
            param.requires_grad = "backbone" not in lowered
        elif scope == "neck_head":
            param.requires_grad = ("neck" in lowered) or ("head" in lowered)
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
