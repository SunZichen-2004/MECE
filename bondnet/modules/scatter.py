import torch


def scatter_sum(values: torch.Tensor, indices: torch.Tensor, output_size: int) -> torch.Tensor:
    output_shape = (output_size,) + values.shape[1:]
    output = values.new_zeros(output_shape)
    output.index_add_(0, indices, values)
    return output
