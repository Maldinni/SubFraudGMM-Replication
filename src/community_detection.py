import os
import leidenalg
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from collections import deque

from utils.parsing import parse_args, load_config
from utils.load_data import ensure_dirs, save_organized_clusters
from utils.organizing import ordenate_clusters, split_clusters


def detect_communities_for_product(product, cfg):
    """Roda a detecção de comunidades (Leiden) para um único produto."""
    paths = cfg["paths"]

    csv_file = os.path.join(paths["processed"], f"{product}.csv")
    graph_file = os.path.join(paths["processed"], "graphs", f"{product}_similarity.graphml")
    cluster_output_directory = os.path.join(paths["processed"], "separated clusters", product)

    if not os.path.exists(graph_file):
        print(f"[{product}] grafo {graph_file} não encontrado, pulando.")
        return
    if not os.path.exists(csv_file):
        print(f"[{product}] csv {csv_file} não encontrado, pulando.")
        return

    print(f'[{product}] loading graph')
    G = ig.Graph.Read(graph_file, format='graphml')

    ids = G.vs['licitacao_id']

    print(f'[{product}] running Leiden community detection for different resolution parameters')
    num_resolution_parameter = cfg['community_detection']['num_resolution_parameter']
    max_resolution_parameter = cfg['community_detection']['max_resolution_parameter']
    min_resolution_parameter = max_resolution_parameter / num_resolution_parameter

    resolution_parameters = np.linspace(min_resolution_parameter,
                                        max_resolution_parameter,
                                        num_resolution_parameter)

    modularity_values = np.zeros(num_resolution_parameter)
    num_unique_clusters = np.zeros(num_resolution_parameter)

    decreasing = deque(np.zeros(5, dtype=bool))

    for i, resolution_parameter in enumerate(resolution_parameters):
        partition = leidenalg.find_partition(
            G,
            leidenalg.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution_parameter)

        modularity_values[i] = G.modularity(partition.membership, weights='weight')
        num_unique_clusters[i] = len(np.unique(partition.membership))

        if i > 0:
            decreasing.popleft()
            decreasing.append(modularity_values[i] < modularity_values[i - 1])

        if all(decreasing):
            print(f'[{product}] modularity is decreasing, breaking')
            break

    best_resolution_parameter = resolution_parameters[np.argmax(modularity_values)]
    partition = leidenalg.find_partition(
        G,
        leidenalg.CPMVertexPartition,
        weights='weight',
        resolution_parameter=best_resolution_parameter)

    id_cluster = dict(zip(ids, partition.membership))

    print(f'[{product}] saving cluster')
    df = pd.read_csv(csv_file)
    df["Cluster ID"] = df["ID"].map(id_cluster)

    clustered_file = os.path.join(paths["processed"], f"{product}_clustered.csv")
    df.to_csv(clustered_file, index=False)

    print(f"[{product}] IDs no grafo: {len(ids)} | IDs no CSV: {len(df)} | "
          f"correspondências: {df['ID'].isin(id_cluster.keys()).sum()}")

    df_organized = ordenate_clusters(pd.read_csv(clustered_file))

    save_organized_clusters(
        df_organized,
        paths["processed"],
        filename=f"{product}_clustered_organized.csv"
    )

    split_clusters(df_organized, cluster_output_directory)

    print(f"[{product}] organização finalizada.")


def main():
    os.chdir('..')  # Change to project root directory
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    for product in cfg["products"]:
        print(f"=== Comunidades: {product} ===")
        detect_communities_for_product(product, cfg)


if __name__ == '__main__':
    main()
