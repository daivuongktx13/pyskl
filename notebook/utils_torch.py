import torch

def weighted_linspace(V: int, count: int, probabilities: torch.FloatTensor, device = 'device'):
    normalized_probs = probabilities / torch.sum(probabilities, dim= -1)
    counts = normalized_probs * count
    rounded_counts = torch.floor(counts).long()
    remaining_count = count - torch.sum(rounded_counts, dim = -1)

    fractional_parts = counts - rounded_counts

    sorted_indices = torch.argsort(fractional_parts, dim = -1, descending=True)

    vertexes_tensor = torch.arange(0, V, device=device, requires_grad= False)

    shift_index = torch.empty((V, count), dtype=torch.long, device=device)

    for i in range(V):
        for j in range(remaining_count[i]):
            rounded_counts[i, sorted_indices[i, j]] += 1

        shift_index[i] = torch.repeat_interleave(vertexes_tensor, rounded_counts[i], dim = 0 )    
    
    return shift_index

def get_shift_indexes(attention_map, channels = 64, device = 'cpu'):

    N, V, _ = attention_map.shape

    shift_indexes = torch.empty((N, V, channels), dtype=torch.long, device=device)
    
    for i, attention_row in enumerate(attention_map):
        V, _ = attention_row.shape
        shift_index = weighted_linspace(V, channels, attention_row, device)
        shift_indexes[i] = shift_index
    
    return shift_indexes
        


