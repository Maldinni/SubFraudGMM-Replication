import os
import leidenalg
import numpy as np
import pandas as pd
import igraph as ig
from pathlib import Path
from collections import deque

from utils.parsing import parse_args, load_config, chdir_to_project_root
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

    # O parâmetro de resolução do CPM não tem escala natural: varremos uma faixa de
    # resoluções (cada uma otimizada pela função de qualidade CPM) e selecionamos a
    # partição que maximiza a MODULARIDADE — uma medida quase livre de escala. Daí o
    # particionamento ser CPM mas o critério de seleção ser modularidade (intencional).
    modularity_values = np.zeros(num_resolution_parameter)

    decreasing = deque(np.zeros(5, dtype=bool))

    last_evaluated = 0
    for i, resolution_parameter in enumerate(resolution_parameters):
        partition = leidenalg.find_partition(
            G,
            leidenalg.CPMVertexPartition,
            weights='weight',
            resolution_parameter=resolution_parameter)

        modularity_values[i] = G.modularity(partition.membership, weights='weight')
        last_evaluated = i

        if i > 0:
            decreasing.popleft()
            decreasing.append(modularity_values[i] < modularity_values[i - 1])

        if all(decreasing):
            print(f'[{product}] modularity is decreasing, breaking')
            break

    # argmax apenas sobre as resoluções de fato avaliadas: com early-stopping o restante
    # do array fica zerado, e um argmax cego escolheria uma resolução nunca testada caso
    # todas as modularidades avaliadas fossem negativas.
    best_idx = int(np.argmax(modularity_values[:last_evaluated + 1]))
    best_resolution_parameter = resolution_parameters[best_idx]
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
    chdir_to_project_root()
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    for product in cfg["products"]:
        print(f"=== Comunidades: {product} ===")
        detect_communities_for_product(product, cfg)


if __name__ == '__main__':
    main()
