import torch
from torch import nn

from fedmox import SpatialTop1MoE, Top1MoE, fedavg, hard_top1_mask, soft_mixture


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
