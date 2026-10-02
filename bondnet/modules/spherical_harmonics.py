import math

import torch


def real_spherical_harmonics(unit_vectors: torch.Tensor, maximum_angular_momentum: int) -> list[torch.Tensor]:
    """Normalized real spherical harmonics through angular momentum two.

    Components use the order [m=0, cosine(1), sine(1), cosine(2), sine(2), ...].
    This real convention makes every positive absolute magnetic order an explicit
    two-component O(2) block.
    """
    if maximum_angular_momentum > 2:
        raise ValueError("This transparent prototype supports maximum angular momentum at most two")
    x_coordinate, y_coordinate, z_coordinate = unit_vectors.unbind(-1)
    values = [unit_vectors.new_full((unit_vectors.shape[0], 1), 1.0 / math.sqrt(4.0 * math.pi))]
    if maximum_angular_momentum >= 1:
        normalization = math.sqrt(3.0 / (4.0 * math.pi))
        values.append(torch.stack((normalization * z_coordinate, normalization * x_coordinate, normalization * y_coordinate), dim=-1))
    if maximum_angular_momentum >= 2:
        zero_normalization = math.sqrt(5.0 / (16.0 * math.pi))
        one_normalization = math.sqrt(15.0 / (4.0 * math.pi))
        two_normalization = math.sqrt(15.0 / (16.0 * math.pi))
        values.append(torch.stack((
            zero_normalization * (3.0 * z_coordinate.square() - 1.0),
            one_normalization * x_coordinate * z_coordinate,
            one_normalization * y_coordinate * z_coordinate,
            two_normalization * (x_coordinate.square() - y_coordinate.square()),
            2.0 * two_normalization * x_coordinate * y_coordinate,
        ), dim=-1))
    return values


def split_magnetic_blocks(harmonic_values: torch.Tensor, angular_momentum: int) -> list[torch.Tensor]:
    blocks = [harmonic_values[:, 0:1]]
    for absolute_magnetic_order in range(1, angular_momentum + 1):
        start = 1 + 2 * (absolute_magnetic_order - 1)
        blocks.append(harmonic_values[:, start : start + 2])
    return blocks


def real_wigner_transport_matrices(
    relative_cartesian_rotations: torch.Tensor,
    angular_momentum: int,
) -> torch.Tensor:
    """Real O(3) representation matrices in this module's harmonic ordering.

    ``relative_cartesian_rotations`` has shape ``[n_pairs, 3, 3]`` and maps
    Cartesian vector components from a source edge frame to a target edge
    frame.  The returned matrix has shape
    ``[n_pairs, 2 * angular_momentum + 1, 2 * angular_momentum + 1]`` and maps
    real spherical components in the same direction.  The implementation is
    analytic for ``l <= 2`` and also handles improper rotations.
    """
    number_of_pairs = relative_cartesian_rotations.shape[0]
    if angular_momentum == 0:
        return relative_cartesian_rotations.new_ones((number_of_pairs, 1, 1))
    if angular_momentum == 1:
        # Cartesian order is [x, y, z], while the real harmonic order is [z, x, y].
        harmonic_to_cartesian = relative_cartesian_rotations.new_tensor([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        return harmonic_to_cartesian @ relative_cartesian_rotations @ harmonic_to_cartesian.transpose(0, 1)
    if angular_momentum != 2:
        raise ValueError("real Wigner transport is implemented for angular momentum zero through two")

    # Y_2,a(x) = x^T basis_matrices[a] x in the exact normalization used above.
    zero_normalization = math.sqrt(5.0 / (16.0 * math.pi))
    one_normalization = math.sqrt(15.0 / (4.0 * math.pi))
    two_normalization = math.sqrt(15.0 / (16.0 * math.pi))
    basis_matrices = relative_cartesian_rotations.new_zeros((5, 3, 3))
    basis_matrices[0, 0, 0] = -zero_normalization
    basis_matrices[0, 1, 1] = -zero_normalization
    basis_matrices[0, 2, 2] = 2.0 * zero_normalization
    basis_matrices[1, 0, 2] = basis_matrices[1, 2, 0] = 0.5 * one_normalization
    basis_matrices[2, 1, 2] = basis_matrices[2, 2, 1] = 0.5 * one_normalization
    basis_matrices[3, 0, 0] = two_normalization
    basis_matrices[3, 1, 1] = -two_normalization
    basis_matrices[4, 0, 1] = basis_matrices[4, 1, 0] = two_normalization
    gram_matrix = torch.einsum("aij,bij->ab", basis_matrices, basis_matrices)
    rotated_basis = torch.einsum(
        "pki,akl,plj->paij",
        relative_cartesian_rotations,
        basis_matrices,
        relative_cartesian_rotations,
    )
    overlaps = torch.einsum("paij,bij->pab", rotated_basis, basis_matrices)
    return overlaps @ torch.linalg.inv(gram_matrix)
