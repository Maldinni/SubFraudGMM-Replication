import os
import igraph as ig
import psutil
from glob import glob
from utils.clustering import construct_knn_graph, check_memory_constraints
from utils.parsing import parse_args, load_config, chdir_to_project_root
from utils.load_data import ensure_dirs, load_embedding_shards


def build_graph_for_product(product, cfg, graph_cfg):
    """Constrói o grafo de similaridade k-NN para um único produto."""
    paths = cfg["paths"]

    shard_directory = os.path.join(paths["raw"], "shards_h5", product)
    graph_directory = os.path.join(paths["processed"], "graphs")
    os.makedirs(graph_directory, exist_ok=True)

    shards_files = glob(os.path.join(shard_directory, '*.h5'))
    graph_file = os.path.join(graph_directory, f'{product}_similarity.graphml')

    if not shards_files:
        print(f"[{product}] nenhum shard .h5 em {shard_directory}, pulando.")
        return

    print(f'[{product}] Loading embeddings...')
    embeddings, texts, ids = load_embedding_shards(shards_files)

    num_points = embeddings.shape[0]
    num_neighbors = graph_cfg['num_neighbors']
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Check if the selected k will fit into memory
    if not check_memory_constraints(num_points, num_neighbors,
                                    available_memory_gb):
        raise MemoryError(
            "Not enough memory for the selected k. Please reduce k or upgrade your hardware."
        )

    print(f'[{product}] Constructing k-NN graph...')
    edges, weights = construct_knn_graph(
        embeddings,
        num_neighbors,
        sim_threshold=graph_cfg.get('sim_threshold', 0.0),
        center=graph_cfg.get('center_embeddings', True),
    )

    print(f'[{product}] Creating igraph Graph object...')
    # n=num_points garante que vértices isolados (sem aresta após o threshold)
    # existam e que os atributos por posição (texts/ids) fiquem alinhados.
    G = ig.Graph(n=num_points, edges=edges, directed=False)
    G.vs['texts'] = texts
    G.vs['licitacao_id'] = ids
    G.es['weight'] = weights

    print(f'[{product}] Saving graph...')
    G.write(graph_file, format='graphml')


def main():
    chdir_to_project_root()
    args = parse_args()
    cfg = load_config(args)

    graph_cfg = cfg["graph_construction"]

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    for product in cfg["products"]:
        print(f"=== Grafo: {product} ===")
        build_graph_for_product(product, cfg, graph_cfg)


if __name__ == '__main__':
    main()
