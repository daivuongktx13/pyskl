import torch

def get_shift_indexes(attention_map, channels = 64, device = 'cuda'):

    N, V, _ = attention_map.shape

    attention_map = attention_map.view(-1, V)

    counts = attention_map * channels
    rounded_counts = torch.floor(counts).long()
    remaining_count = channels - torch.sum(rounded_counts, dim = -1)
    fractional_parts = counts - rounded_counts

    sorted_indices = torch.argsort(fractional_parts, dim = -1, descending=True)
    fake_mask = torch.arange(V, device=device).reshape(1, V).repeat(N * V, 1) < remaining_count.unsqueeze(1)

    real_mask = torch.gather(fake_mask, 1, sorted_indices.argsort())

    zero_mask = torch.zeros((N * V, V), dtype=torch.long, device=device)
    one_mask = torch.ones((N * V, V), dtype=torch.long, device=device)
    final_counts = zero_mask.masked_scatter(real_mask, one_mask) + rounded_counts

    vertexes_tensor = torch.arange(0, V, device=device).repeat(N * V)

    shift_indexes = torch.repeat_interleave(vertexes_tensor, final_counts.view(-1)).view(N, V, -1)
    
    position_indexes = torch.arange(channels, dtype=torch.long, device=device).reshape((1, 1, -1)).repeat(N, V, 1)
    print(shift_indexes * channels + position_indexes)

    return shift_indexes * channels + position_indexes

