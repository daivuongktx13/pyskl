import numpy as np

def weighted_linspace(V: int, count: int, probabilities: np.array):
    normalized_probs = probabilities / np.sum(probabilities, axis= -1)
    counts = normalized_probs * count
    rounded_counts = np.floor(counts).astype(int)
    remaining_count = count - np.sum(rounded_counts, axis = -1)

    fractional_parts = counts - rounded_counts

    sorted_indices = np.argsort(fractional_parts, axis = -1)[:, ::-1]
    vertexes_tensor = np.arange(0, V)

    shift_index = np.zeros((V, count), dtype=int)

    for i in range(V):
        for j in range(remaining_count[i]):
            rounded_counts[i, sorted_indices[i, j]] += 1

        shift_index[i] = np.repeat(vertexes_tensor, rounded_counts[i], axis = 0 )    
    
    return shift_index

def get_shift_indexes(attention_map, channels = 64):

    N, V, _ = attention_map.shape

    shift_indexes = np.zeros((N, V,channels), dtype=int)
    
    for i, attention_row in enumerate(attention_map):
        V, _ = attention_row.shape
        shift_index = weighted_linspace(V, channels, attention_row)
        shift_indexes[i] = shift_index
    
    return shift_indexes
        


