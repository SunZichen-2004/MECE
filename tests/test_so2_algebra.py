import torch

from bondnet import AtomicData, BondNet
from bondnet.modules.edge_cluster_expansion import multiply_so2_blocks
from bondnet.modules.spherical_harmonics import real_spherical_harmonics, real_wigner_transport_matrices


def test_real_wigner_transport_matches_direct_harmonic_evaluation() -> None:
    torch.manual_seed(23)
    rotations, _ = torch.linalg.qr(torch.randn(7, 3, 3, dtype=torch.float64))
    determinant_sign = torch.linalg.det(rotations).sign().view(-1, 1, 1)
    rotations[:, :, 0:1] *= determinant_sign
    source_directions = torch.randn(7, 3, dtype=torch.float64)
    source_directions /= torch.linalg.vector_norm(source_directions, dim=-1, keepdim=True)
    target_directions = torch.einsum("pij,pj->pi", rotations, source_directions)
    source_harmonics = real_spherical_harmonics(source_directions, 2)
    target_harmonics = real_spherical_harmonics(target_directions, 2)
    for angular_momentum in range(3):
        transported = torch.einsum(
            "pab,pb->pa",
            real_wigner_transport_matrices(rotations, angular_momentum),
            source_harmonics[angular_momentum],
        )
        torch.testing.assert_close(transported, target_harmonics[angular_momentum], rtol=1.0e-12, atol=1.0e-12)


def test_so2_product_has_sum_and_difference_orders() -> None:
    angle = torch.tensor(0.37, dtype=torch.float64)
    left = torch.stack((torch.cos(2 * angle), torch.sin(2 * angle))).view(1, 2, 1)
    right = torch.stack((torch.cos(angle), torch.sin(angle))).view(1, 2, 1)
    products = dict(multiply_so2_blocks(left, 2, right, 1))
    expected_order_three = torch.stack((torch.cos(3 * angle), torch.sin(3 * angle))).view(1, 2, 1)
    expected_order_one = torch.stack((torch.cos(angle), torch.sin(angle))).view(1, 2, 1)
    torch.testing.assert_close(products[3], expected_order_three)
    torch.testing.assert_close(products[1], expected_order_one)


def test_edge_reversal_gives_one_shared_bond_energy() -> None:
    torch.manual_seed(29)
    model = BondNet(number_of_channels=4, maximum_angular_momentum=1, number_of_interactions=1)
    data = AtomicData(
        torch.tensor([1, 6, 8]),
        torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.3, 0.2]]),
        cutoff_radius=model.cutoff_radius,
    )
    result = model(data)
    sender = data.edge.sender_atom_indices
    receiver = data.edge.receiver_atom_indices
    for edge_index in range(sender.numel()):
        reverse_index = torch.where(
            (sender == receiver[edge_index]) & (receiver == sender[edge_index])
        )[0].item()
        torch.testing.assert_close(result["edge_energy"][edge_index], result["edge_energy"][reverse_index])
    torch.testing.assert_close(result["energy"].sum(), result["edge_energy"].sum())
