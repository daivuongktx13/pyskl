import torch

# def weighted_linspace(V: int, count: int, probabilities: torch.FloatTensor, device = 'device'):
#     normalized_probs = probabilities / torch.sum(probabilities, dim= -1)
#     counts = normalized_probs * count
#     rounded_counts = torch.floor(counts).long()
#     remaining_count = count - torch.sum(rounded_counts, dim = -1)

#     fractional_parts = counts - rounded_counts

#     sorted_indices = torch.argsort(fractional_parts, dim = -1, descending=True)

#     vertexes_tensor = torch.arange(0, V, device=device, requires_grad= False)

#     shift_index = torch.empty((V, count), dtype=torch.long, device=device)

#     for i in range(V):
#         for j in range(remaining_count[i]):
#             rounded_counts[i, sorted_indices[i, j]] += 1

#         shift_index[i] = torch.repeat_interleave(vertexes_tensor, rounded_counts[i], dim = 0 ) 
    
#     return shift_index

# def get_shift_indexes(attention_map, channels = 64, device = 'cpu'):

#     N, V, _ = attention_map.shape

#     shift_indexes = torch.empty((N, V, channels), dtype=torch.long, device=device)
#     position_indexes = torch.arange(channels, dtype=torch.long, device=device, requires_grad=False).reshape((1, 1, -1)).repeat(N, V, 1)
    
#     for i, attention_row in enumerate(attention_map):
#         V, _ = attention_row.shape
#         shift_index = weighted_linspace(V, channels, attention_row, device)
#         shift_indexes[i] = shift_index
    
#     return position_indexes +  shift_indexes * channels
        

    
def get_shift_indexes(attention_map, channels = 64, device = 'cpu'):

    N, V, _ = attention_map.shape

    attention_map = attention_map.view(-1, V)

    counts = attention_map * channels
    rounded_counts = torch.floor(counts).long()
    remaining_count = channels - torch.sum(rounded_counts, dim = -1)
    fractional_parts = counts - rounded_counts

    sorted_indices = torch.argsort(fractional_parts, dim = -1, descending=True)
    fake_mask = torch.lt(torch.arange(V, device=device).reshape(1, V).repeat(N * V, 1), remaining_count.unsqueeze(1))

    real_mask = torch.gather(fake_mask, 1, sorted_indices.argsort())

    zero_mask = torch.zeros((N * V, V), dtype=torch.long, device=device)
    one_mask = torch.ones((N * V, V), dtype=torch.long, device=device)
    final_counts = zero_mask.masked_scatter(real_mask, one_mask) + rounded_counts

    vertexes_tensor = torch.arange(0, V, device=device).repeat(N * V)

    shift_indexes = torch.repeat_interleave(vertexes_tensor, final_counts.view(-1)).view(N, V, -1)
    
    position_indexes = torch.arange(channels, dtype=torch.long, device=device).reshape((1, 1, -1)).repeat(N, V, 1)

    return torch.sum(shift_indexes * channels + position_indexes)


def get_shift_indexes_with_test_case(attention_map, channels = 64, device = 'cpu'):

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

    for case in range(N * V):
        from_sorted = sorted_indices[case][:remaining_count[case]].sort().values
        from_real_mask = torch.where(real_mask[case] == True)[0]
        assert all(torch.eq(from_real_mask, from_sorted))

    assert all(torch.sum(final_counts, dim = -1) == channels)

    vertexes_tensor = torch.arange(0, V, device=device, requires_grad= False).repeat(N * V)

    shift_indexes = torch.repeat_interleave(vertexes_tensor, final_counts.view(-1)).view(N, V, -1)
    
    position_indexes = torch.arange(channels, dtype=torch.long, device=device, requires_grad=False).reshape((1, 1, -1)).repeat(N, V, 1)

    return shift_indexes * channels + position_indexes




    
        

