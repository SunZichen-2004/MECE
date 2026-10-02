import torch

from .scatter import scatter_sum


def construct_edge_frames(
    unit_displacement_vectors: torch.Tensor,
    sender_atom_indices: torch.Tensor,
    receiver_atom_indices: torch.Tensor,
    number_of_atoms: int,
) -> torch.Tensor:
    """Construct right-handed, environment-covariant edge frames.

    The third axis follows the directed bond. The first axis is the projected
    first moment of the sender and receiver environments. A Cartesian fallback
    is used only for exactly degenerate environments.
    """
    environment_moments = scatter_sum(unit_displacement_vectors, sender_atom_indices, number_of_atoms)
    transverse_candidates = environment_moments[sender_atom_indices] + environment_moments[receiver_atom_indices]
    third_axes = unit_displacement_vectors
    transverse_candidates = transverse_candidates - (
        transverse_candidates * third_axes
    ).sum(dim=-1, keepdim=True) * third_axes
    candidate_norms = torch.linalg.vector_norm(transverse_candidates, dim=-1, keepdim=True)
    first_reference = torch.zeros_like(third_axes)
    first_reference[:, 0] = 1.0
    second_reference = torch.zeros_like(third_axes)
    second_reference[:, 1] = 1.0
    use_second = (third_axes[:, 0].abs() > 0.9).unsqueeze(-1)
    reference_axes = torch.where(use_second, second_reference, first_reference)
    fallback = reference_axes - (reference_axes * third_axes).sum(dim=-1, keepdim=True) * third_axes
    transverse_candidates = torch.where(candidate_norms > 1.0e-7, transverse_candidates, fallback)
    first_axes = transverse_candidates / torch.linalg.vector_norm(transverse_candidates, dim=-1, keepdim=True).clamp_min(1.0e-12)
    second_axes = torch.linalg.cross(third_axes, first_axes, dim=-1)
    return torch.stack((first_axes, second_axes, third_axes), dim=-2)


def rotate_vectors_to_edge_frames(vectors: torch.Tensor, edge_frames: torch.Tensor) -> torch.Tensor:
    return torch.einsum("eij,ej->ei", edge_frames, vectors)


def relative_frame_rotations(
    target_edge_frames: torch.Tensor,
    source_edge_frames: torch.Tensor,
) -> torch.Tensor:
    """Return vector-component transport from source edge frames to target frames.

    Both inputs contain frame axes as rows in the global frame and have shape
    ``[n_edge_pairs, 3, 3]``.  The result has shape ``[n_edge_pairs, 3, 3]`` and
    acts on Cartesian components written in the source frame:

    ``vector_in_target = relative_rotation @ vector_in_source``.
    """
    return target_edge_frames @ source_edge_frames.transpose(-1, -2)
