"""
Distinção entre clusters semânticos — etapa opcional após cluster_definition.

Para cada cluster, o pipeline já calculou (em cluster_definition.convert_product) o
"Cluster Mais Similar" e a "Similaridade". Aqui pegamos esse par de clusters quase
idênticos e pedimos ao auditor LLM que explique POR QUE eles são distintos, em termos
relevantes para auditoria de licitações.

Por que isso importa no contexto atual (vindo de um sistema anterior de chamados, onde
servia para separar tópicos de suporte): dois grupos de licitações altamente similares —
mesmo objeto/município, especificação quase igual — que ainda assim formam clusters
separados são exatamente o sinal de splitting artificial (fracionamento, cartel rotativo,
empresa de fachada). Saber a diferença mínima entre eles (fornecedor? faixa de preço?
janela temporal?) direciona a diligência do auditor.

Roda DEPOIS de cluster_definition (que produz {produto}_clusters_defined.csv) e itera por
produto, como o restante do pipeline.
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from utils.parsing import (
    parse_args,
    load_config,
    chdir_to_project_root,
    parse_distinction_file,
    extract_cluster_id,
)
from utils.hypersphere import get_centroids, get_representative_articles
from utils.load_data import ensure_dirs, load_embedding_shards, align_to_df

from cluster_definition import build_texto_cluster, truncate


DISTINCTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    Você é um auditor fiscal sênior especializado em licitações públicas brasileiras, com foco em
    conluio (cartel de licitação), direcionamento de edital, superfaturamento e fracionamento da despesa.

    Dois grupos de licitações (CLUSTER A e CLUSTER B) ficaram semanticamente MUITO próximos — quase
    o mesmo objeto. Sua tarefa é explicar a diferença mínima que os separa. Em auditoria, dois grupos
    quase idênticos que ainda assim se distinguem podem indicar splitting artificial: fracionamento
    para fugir de modalidade, rodízio de vencedores (cartel) ou empresas de fachada espelhando um
    fornecedor real.

    Compare priorizando os eixos relevantes para fraude:
    - Fornecedor / CNPJ (mesma empresa ou empresas distintas espelhadas?)
    - Faixa de preço unitário e sobrepreço relativo
    - Taxa de vitória (win) e concentração de vencedores (unique)
    - Especificação/objeto (direcionamento sutil?)
    - Padrão temporal e unidade gestora

    Baseie-se SOMENTE nos campos presentes nos registros. Não invente dados.
    """,
        ),
        (
            "user",
            """
    ## TAREFA: DISTINÇÃO ENTRE DOIS CLUSTERS DE LICITAÇÕES

    Cada registro traz métricas de risco já calculadas (win, unique, Risk_Mean/Std/Max, Rank_Mean).

    ### CLUSTER A (id {label_a})
    {cluster_a}

    ### CLUSTER B (id {label_b}) — o mais similar a A
    {cluster_b}

    ---

    Responda EXATAMENTE no formato abaixo, sem texto fora dele.

    DIFERENÇAS:
    - <diferença 1, citando o campo que a sustenta>
    - <diferença 2, citando o campo que a sustenta>
    - <diferença 3, citando o campo que a sustenta>

    RELEVÂNCIA PARA AUDITORIA:
    <1–2 frases: a diferença sugere splitting artificial (fracionamento/cartel/fachada) ou são
    aquisições genuinamente distintas? Classifique: 🔴 ALTO | 🟡 MÉDIO | 🟢 BAIXO>
    """,
        ),
    ]
)


def distinguish_clusters_for_product(product, cfg, distinction_chain):
    """Gera a distinção textual entre cada cluster e seu cluster mais similar."""
    paths = cfg["paths"]

    clustered_file = os.path.join(paths["processed"], f"{product}_clustered.csv")
    defined_file = os.path.join(paths["processed"], f"{product}_clusters_defined.csv")
    shard_directory = os.path.join(paths["raw"], "shards_h5", product)
    output_directory = os.path.join(paths["output"], product, "cluster_distinctions")

    if not os.path.exists(clustered_file):
        print(f"[{product}] {clustered_file} não encontrado, pulando.")
        return
    if not os.path.exists(defined_file):
        print(f"[{product}] {defined_file} não encontrado (rode cluster_definition antes), pulando.")
        return

    os.makedirs(output_directory, exist_ok=True)

    df = pd.read_csv(clustered_file)
    df = df[df["Cluster ID"].notna()].copy()
    df["Cluster ID"] = df["Cluster ID"].astype(int)

    if df["Cluster ID"].nunique() < 2:
        print(f"[{product}] menos de 2 clusters, nada a distinguir.")
        return

    df["texto_cluster"] = build_texto_cluster(df)

    # Mapa {cluster_id: cluster_mais_similar} produzido por cluster_definition.
    defined_df = pd.read_csv(defined_file)
    most_similar = dict(
        zip(defined_df["Cluster ID"].astype(int), defined_df["Cluster Mais Similar"].astype(int))
    )

    shards_files = glob(os.path.join(shard_directory, "*.h5"))
    embeddings, texts, ids = load_embedding_shards(shards_files)
    aligned_embeddings, _ = align_to_df(embeddings, ids, df)

    labels = df["Cluster ID"].values
    cluster_texts = df["texto_cluster"].values
    centroids = get_centroids(aligned_embeddings, labels)

    # Centróides são indexados por rótulo ordenado; Leiden/CPM pode gerar rótulos não
    # contíguos, então mapeamos cada rótulo à sua linha em vez de assumir label == índice.
    unique_labels = np.unique(labels)
    label_to_row = {int(label): row for row, label in enumerate(unique_labels)}

    k = cfg["definition"]["max_article_number"]

    print(f"[{product}] distinguindo {len(unique_labels)} clusters...")

    for label in unique_labels:
        label = int(label)
        output_path = os.path.join(output_directory, f"cluster_{label}_distinction.txt")

        # Retomada: pula clusters já distinguidos.
        if os.path.exists(output_path):
            continue

        similar_label = most_similar.get(label)
        if similar_label is None or similar_label not in label_to_row:
            print(f"[{product}] cluster {label} sem cluster similar válido, ignorando.")
            continue

        mask_a = labels == label
        mask_b = labels == similar_label

        # Registros representativos de cada cluster (mais próximos do PRÓPRIO centróide).
        reps_a = get_representative_articles(
            centroids[label_to_row[label]],
            aligned_embeddings[mask_a],
            cluster_texts[mask_a],
            min(k, int(mask_a.sum())),
        )
        reps_b = get_representative_articles(
            centroids[label_to_row[similar_label]],
            aligned_embeddings[mask_b],
            cluster_texts[mask_b],
            min(k, int(mask_b.sum())),
        )

        chain_input = {
            "label_a": label,
            "label_b": similar_label,
            "cluster_a": "\n\n".join(truncate(a) for a in reps_a),
            "cluster_b": "\n\n".join(truncate(b) for b in reps_b),
        }

        raw_output = distinction_chain.invoke(chain_input)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw_output.content)

        print(f"[{product}] cluster {label} vs {similar_label} salvo.")


def main():
    chdir_to_project_root()
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    llm = ChatOllama(
        model=cfg["llm"].get("model_name", "qwen2.5:7b"),
        temperature=cfg["llm"]["temperature"],
    )
    distinction_chain = DISTINCTION_PROMPT | llm

    for product in cfg["products"]:
        print(f"=== Distinção de clusters: {product} ===")
        distinguish_clusters_for_product(product, cfg, distinction_chain)


def convert_product(product, cfg):
    """Anexa as distinções textuais (.txt) ao CSV de definições do produto."""
    paths = cfg["paths"]

    defined_file = os.path.join(paths["processed"], f"{product}_clusters_defined.csv")
    distinction_dir = os.path.join(paths["output"], product, "cluster_distinctions")
    output_file = os.path.join(paths["processed"], f"{product}_clusters_distinguished.csv")

    if not os.path.exists(defined_file):
        print(f"[{product}] {defined_file} não encontrado, pulando.")
        return

    distinction_files = glob(os.path.join(distinction_dir, "*.txt"))
    if not distinction_files:
        print(f"[{product}] nenhuma distinção em {distinction_dir}, pulando.")
        return

    records = []
    for file in distinction_files:
        records.append({
            "Cluster ID": extract_cluster_id(file),
            "Diferencas": parse_distinction_file(file),
        })

    dist_df = pd.DataFrame(records)
    cluster_df = pd.read_csv(defined_file)

    final_df = cluster_df.merge(dist_df, on="Cluster ID", how="left")
    final_df.to_csv(output_file, index=False)

    print(f"[{product}] dataset com distinções criado: {output_file}")


def json_converter():
    chdir_to_project_root()
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    for product in cfg["products"]:
        print(f"=== Conversão de distinções: {product} ===")
        convert_product(product, cfg)


if __name__ == "__main__":
    main()
    json_converter()
