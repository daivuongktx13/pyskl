import numpy as np
import torch

# def weighted_linspace(count, probabilities):
#     normalized_probs = np.array(probabilities) / np.sum(probabilities)
#     counts = normalized_probs * count
#     rounded_counts = np.round(counts)
#     total_rounded = int(np.sum(rounded_counts))

#     if total_rounded < count:
#         remaining_count = count - total_rounded
#         sorted_indices = np.argsort(counts - rounded_counts)[::-1]
#         for i in range(remaining_count):
#             rounded_counts[sorted_indices[i]] += 1
#     elif total_rounded > count:
#         remaining_count = total_rounded - count
#         sorted_indices = np.argsort(rounded_counts - counts)[::-1]
#         for i in range(remaining_count):
#             rounded_counts[sorted_indices[i]] -= 1

#     # result = []
#     # for i in range(len(rounded_counts)):
#     #     result.extend([i] * int(rounded_counts[i]))

#     return rounded_counts

# def batch_weighted_linspace(N, count, probabilities):
#     normalized_probs = np.array(probabilities) / np.sum(probabilities, axis=1, keepdims=True)
#     counts = normalized_probs * count
#     rounded_counts = np.floor(counts).astype(int)
#     remaining_count = count - np.sum(rounded_counts, axis=1)

#     fractional_parts = counts - rounded_counts
#     sorted_indices = np.argsort(fractional_parts, axis=1)[:, ::-1]
#     for i in range(N):
#         for j in range(remaining_count[i]):
#             rounded_counts[i, sorted_indices[i, j]] += 1

#     # result = []
#     # for i in range(N):
#     #     row = []
#     #     for j in range(len(rounded_counts[i])):
#     #         row.extend([j] * rounded_counts[i, j])
#     #     result.append(row)

#     return rounded_counts

def weighted_linspace(V: int, count: int, probabilities: torch.FloatTensor, device = 'cpu'):
    normalized_probs = probabilities / torch.sum(probabilities, dim = -1)
    counts = normalized_probs * count
    rounded_counts = torch.floor(counts).long()
    remaining_count = count - torch.sum(rounded_counts, dim = -1)

    fractional_parts = counts - rounded_counts

    sorted_indices = torch.argsort(fractional_parts, dim = -1, descending=True)
    vertexes_tensor = torch.arange(0, V, device=device, requires_grad= False)

    shift_index = torch.zeros((V, count), dtype=torch.long, device=device)

    # shift_index = []
    for i in range(V):
        for j in range(remaining_count[i]):
            rounded_counts[i, sorted_indices[i, j]] += 1

        shift_index[i] = torch.repeat_interleave(vertexes_tensor, rounded_counts[i], dim = 0 )
        
        # shift_index.append(torch.repeat_interleave(vertexes_tensor, rounded_counts[i], dim = 0 ))
    
    
    return shift_index

def get_shift_indexes(attention_map, channel = 64, device = 'cpu'):

    shift_indexes = []
    for attention_row in attention_map:
        V, _ = attention_row.shape
        shift_index = weighted_linspace(V, channel, attention_row, device)
        shift_indexes.append(shift_index)
    
    return torch.stack(shift_indexes, dim = 0)
        


