import torch
from torch import nn

from fedmox import (
    PSSFLRunner,
    SpatialTop1MoE,
    Top1MoE,
    clone_selected_state_dict,
    fedavg,
    hard_top1_mask,
    soft_mixture,
)


def test_hard_top1_mask_is_one_hot():
    logits = torch.tensor([[[1.0, 5.0], [3.0, 2.0]]])
    mask = hard_top1_mask(logits, expert_dim=1)
    assert torch.equal(mask.sum(dim=1), torch.ones(1, 2))


def test_fedavg_and_soft_mixture_match_paper_equations():
    c1 = {"w": torch.tensor([1.0, 3.0])}
    c2 = {"w": torch.tensor([3.0, 7.0])}
    avg = fedavg([c1, c2], [1, 3])
    assert torch.allclose(avg["w"], torch.tensor([2.5, 6.0]))

    server = {"w": torch.tensor([10.0, 10.0])}
    mixed = soft_mixture(server, avg, alpha=0.25)
    assert torch.allclose(mixed["w"], torch.tensor([4.375, 7.0]))


def test_non_floating_state_is_copied_not_averaged():
    c1 = {"w": torch.tensor([1.0]), "step": torch.tensor(3, dtype=torch.long)}
    c2 = {"w": torch.tensor([3.0]), "step": torch.tensor(9, dtype=torch.long)}
    avg = fedavg([c1, c2], [1, 1])
    assert torch.allclose(avg["w"], torch.tensor([2.0]))
    assert avg["step"].item() == 3


def test_spatial_top1_moe_preserves_resolution():
    moe = SpatialTop1MoE(3, lambda: nn.Conv2d(3, 5, kernel_size=1), num_experts=4)
    low = moe(torch.randn(2, 3, 9, 16))
    high = moe(torch.randn(2, 3, 18, 32))
    assert low.shape == (2, 5, 9, 16)
    assert high.shape == (2, 5, 18, 32)


def test_fixed_top1_moe_handles_roi_vectors():
    moe = Top1MoE(8, lambda: nn.Linear(8, 4), num_experts=3)
    out = moe(torch.randn(2, 7, 8))
    assert out.shape == (2, 7, 4)


def test_router_gets_gradient_with_hard_forward_straight_through_backward():
    moe = Top1MoE(4, lambda: nn.Linear(4, 2), num_experts=3)
    out = moe(torch.randn(5, 4)).sum()
    out.backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum() > 0


def test_federated_state_excludes_backbone_by_default():
    model = nn.Sequential()
    model.add_module("backbone", nn.Linear(2, 2))
    model.add_module("roi_head", nn.Linear(2, 1))
    state = clone_selected_state_dict(model.state_dict())
    assert all("backbone" not in key for key in state)
    assert any("roi_head" in key for key in state)


def test_pssfl_runner_updates_head_without_overwriting_backbone():
    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(1, 1, bias=False)
            self.roi_head = nn.Linear(1, 1, bias=False)

    model = TinyDetector()
    model.backbone.weight.data.fill_(10.0)
    model.roi_head.weight.data.fill_(1.0)
    runner = PSSFLRunner(model, client_ids=[0, 1, 2], seed=0, soft_mixture_alpha=0.5)
    assert runner.frozen_backbone_parameters == 1
    assert model.backbone.weight.requires_grad is False

    def warmup_train(model, epochs):
        model.roi_head.weight.data.fill_(2.0)

    def client_train(client_id, state, epochs):
        updated = {key: value.clone() + float(client_id + 1) for key, value in state.items()}
        from fedmox import ClientUpdate

        return ClientUpdate(client_id=client_id, sample_count=1, state_dict=updated)

    def server_train(model, epochs):
        model.roi_head.weight.data.add_(1.0)

    runner.run(warmup_train=warmup_train, client_train=client_train, server_train=server_train, rounds=1)
    assert torch.allclose(model.backbone.weight, torch.tensor([[10.0]]))
    assert not torch.allclose(model.roi_head.weight, torch.tensor([[2.0]]))


def test_pssfl_runner_requires_explicit_alpha():
    class TinyDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.roi_head = nn.Linear(1, 1, bias=False)

    model = TinyDetector()
    runner = PSSFLRunner(model, client_ids=[0], seed=0)

    def warmup_train(model, epochs):
        pass

    def client_train(client_id, state, epochs):
        from fedmox import ClientUpdate

        return ClientUpdate(client_id=client_id, sample_count=1, state_dict=state)

    def server_train(model, epochs):
        pass

    try:
        runner.run(warmup_train=warmup_train, client_train=client_train, server_train=server_train, rounds=1)
    except ValueError as exc:
        assert "soft_mixture_alpha must be specified" in str(exc)
    else:
        raise AssertionError("missing soft_mixture_alpha should fail")
