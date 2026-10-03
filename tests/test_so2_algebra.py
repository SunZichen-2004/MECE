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


def test_vectorized_so2_product_matches_pairwise_blocks() -> None:
    torch.manual_seed(31)
    from bondnet.modules.edge_cluster_expansion import so2_contract

    number_of_items = 5
    number_of_channels = 3
    maximum_angular_momentum = 2
    orders = []
    for absolute_order in range(maximum_angular_momentum + 1):
        components = 1 if absolute_order == 0 else 2
        orders.append(torch.randn(number_of_items, components, number_of_channels, dtype=torch.float64))
    left = torch.zeros(maximum_angular_momentum + 1, number_of_items, 2, number_of_channels, dtype=torch.float64)
    right = torch.zeros_like(left)
    for absolute_order, values in enumerate(orders):
        left[absolute_order, :, : values.shape[1]] = values
        right[absolute_order, :, : values.shape[1]] = torch.randn_like(values)
    expected = torch.zeros_like(left)
    for left_order in range(maximum_angular_momentum + 1):
        for right_order in range(maximum_angular_momentum + 1):
            left_values = left[left_order, :, : 1 if left_order == 0 else 2]
            right_values = right[right_order, :, : 1 if right_order == 0 else 2]
            for output_order, product in multiply_so2_blocks(left_values, left_order, right_values, right_order):
                if output_order <= maximum_angular_momentum:
                    width = product.shape[1]
                    expected[output_order, :, :width] += product
    actual = so2_contract(left, right)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_so2_product_has_sum_and_difference_orders() -> None:
    angle = torch.tensor(0.37, dtype=torch.float64)
    left = torch.stack((torch.cos(2 * angle), torch.sin(2 * angle))).view(1, 2, 1)
    right = torch.stack((torch.cos(angle), torch.sin(angle))).view(1, 2, 1)
    products = dict(multiply_so2_blocks(left, 2, right, 1))
    expected_order_three = torch.stack((torch.cos(3 * angle), torch.sin(3 * angle))).view(1, 2, 1)
    expected_order_one = torch.stack((torch.cos(angle), torch.sin(angle))).view(1, 2, 1)
    torch.testing.assert_close(products[3], expected_order_three)
    torch.testing.assert_close(products[1], expected_order_one)


def test_total_energy_is_sum_of_bond_layer_energies() -> None:
    torch.manual_seed(29)
    model = BondNet(number_of_channels=4, maximum_angular_momentum=1, number_of_interactions=1)
    data = AtomicData(
        torch.tensor([1, 6, 8]),
        torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.2, 0.0], [0.1, 1.3, 0.2]]),
        cutoff_radius=model.cutoff_radius,
    )
    result = model(data)
    assert result["bond_energy"].shape == data.edge.sender_atom_indices.shape
    torch.testing.assert_close(result["energy"].sum(), result["bond_energy"].sum())
    model.atomic_energy_table[1] = 0.4
    model.atomic_energy_table[6] = -1.5
    model.atomic_energy_table[8] = 2.0
    result = model(data)
    atomic_reference = model.atomic_energy_table[data.atomic_numbers].sum()
    torch.testing.assert_close(result["energy"].sum(), result["bond_energy"].sum() + atomic_reference)
