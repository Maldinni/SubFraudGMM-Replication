import tomllib
import faiss
from tqdm import tqdm
import numpy as np

def estimate_memory_usage(num_points, k):
    """
    Estimate the memory usage for storing k-NN results.
 
    Parameters:
    - num_points: int
    - k: int
 
    Returns:
    - memory_usage_bytes: int
    """
    bytes_per_edge = 12  # 4 bytes for index, 8 bytes for distance
    total_edges = num_points * k
    memory_usage_bytes = total_edges * bytes_per_edge
    return memory_usage_bytes


def check_memory_constraints(num_points, k, available_memory_gb):
    """
    Check if the k-NN computation will fit into available memory.
 
    Parameters:
    - num_points: int
    - k: int
    - available_memory_gb: float
 
    Returns:
    - fits_in_memory: bool
    """
    memory_usage_bytes = estimate_memory_usage(num_points, k)
    memory_usage_gb = memory_usage_bytes / (1024**3)
    if memory_usage_gb > available_memory_gb * 0.8:
        print(
            f"Estimated memory usage ({memory_usage_gb:.2f} GB) exceeds 80% of available memory."
        )
        return False
    else:
        print(f"Estimated memory usage: {memory_usage_gb:.2f} GB")
        return True


def construct_knn_graph(embeddings, k, sim_threshold=0.0, center=True):
    """
    Construct the symmetric k-NN graph using FAISS.

    Parameters:
    - embeddings: np.array
    - k: int
    - sim_threshold: float, similaridade mínima (sobre os vetores centralizados)
      para manter uma aresta. Deve ser >= 0 para garantir pesos positivos no Leiden.
    - center: bool, se True subtrai a média dos embeddings antes de normalizar,
      removendo o componente comum do texto (boilerplate) e espalhando as
      similaridades — tornando o threshold efetivo.

    Returns:
    - edges: list of tuples
    - weights: list of floats
    """
    if center:
        embeddings = embeddings - embeddings.mean(axis=0, keepdims=True)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12  # evita divisão por zero em vetores nulos pós-centralização
    embeddings = embeddings / norms
    num_points, dim = embeddings.shape

    # Build the FAISS index
    index = faiss.IndexFlatIP(
        dim
    )  # Inner product is equivalent to cosine similarity after normalization
    index.add(embeddings)

    # Perform k-NN search
    print("Performing k-NN search...")
    distances, indices = index.search(embeddings, k)

    # Collect edges and weights (apenas arestas acima do threshold de similaridade)
    print(f"Constructing edge list (sim_threshold={sim_threshold})...")
    edge_set = set()

    for i in tqdm(range(num_points)):
        for neighbor_idx, distance in zip(indices[i], distances[i]):
            if neighbor_idx != i and distance >= sim_threshold:
                edge = tuple(sorted((i, neighbor_idx)))
                edge_set.add((edge, float(distance)))

    # Symmetrize the graph
    print("Symmetrizing the graph...")
    # Create a mapping from node to its neighbors
    neighbor_dict = {}
    for (i, j), weight in edge_set:
        neighbor_dict.setdefault(i, set()).add((j, weight))
        neighbor_dict.setdefault(j, set()).add((i, weight))

    # Build the symmetric edge list
    symmetric_edges = set()
    symmetric_weights = []
    for i in tqdm(range(num_points)):
        neighbors = neighbor_dict.get(i, set())
        for j, weight in neighbors:
            if i < j:
                symmetric_edges.add((i, j))
                symmetric_weights.append(weight)

    edges = list(symmetric_edges)
    weights = symmetric_weights

    return edges, weights