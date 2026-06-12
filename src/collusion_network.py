"""
Rede de conluio entre fornecedores — etapa estrutural final (cross-cluster).

As etapas anteriores analisam cada cluster semântico isoladamente. Aqui damos o zoom-out:
montamos um grafo fornecedor↔fornecedor sobre o produto INTEIRO e rodamos detecção de
comunidades (Leiden) para revelar ANÉIS de fornecedores — candidatos a cartel — que se
espalham por vários clusters e municípios.

MODELAGEM E SUA LIMITAÇÃO:
O dataset contém apenas registros vencedores (auction_winner_flag ≡ 1; sem dados dos
perdedores), então NÃO é possível um grafo de co-licitação direta (quem disputou contra
quem). Em vez disso, ligamos dois fornecedores quando eles vencem nas MESMAS unidades
gestoras — o footprint de mercado compartilhado. Fornecedores que dividem muitas UGs, em
mercados onde a proporção de vencedores distintos (unique) é baixa, são o substrato de um
cartel de rodízio — exatamente o padrão da Operação Patrola. É uma inferência por
footprint, não prova de conluio.

Roda por produto, como o restante do pipeline.
"""

import os
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
from glob import glob
from itertools import combinations
from collections import Counter

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from utils.parsing import (
    parse_args,
    load_config,
    chdir_to_project_root,
    extract_cluster_id,
    clean_llm_text,
)
from utils.load_data import ensure_dirs


SUPPLIER_ID_COL = "CPF/CNPJ Participante Cotacao"
SUPPLIER_NAME_COL = "nomeParticipante"
UNIT_COL = "ID UnidadeGestora"


RING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    Você é um auditor fiscal sênior em licitações públicas brasileiras, especializado em cartel
    de licitação (conluio) e rodízio de vencedores. Recebe um ANEL de fornecedores que vencem
    recorrentemente nas mesmas unidades gestoras (mesmo mercado municipal).

    Interprete os números sob a ótica de risco:
    - Muitas unidades gestoras compartilhadas + baixa proporção de vencedores distintos (unique)
      → forte indício de rodízio de vencedores (cartel).
    - Um fornecedor concentra quase todas as vitórias e os demais raramente vencem → possível
      dominador com empresas de fachada (participação fictícia).
    - Risco GMM (Risk_Mean) alto e/ou presença de fraudes confirmadas reforçam o alerta.
    - Atenção a falsos positivos: um único fornecedor regional especializado atuando em muitos
      municípios pode ser legítimo — pondere.

    Use SOMENTE os números fornecidos. Não invente fornecedores nem valores.
    """,
        ),
        (
            "user",
            """
    ## TAREFA: CARACTERIZAÇÃO DE ANEL DE FORNECEDORES

    {ring_summary}

    ---

    Responda EXATAMENTE no formato abaixo, sem texto fora dele.

    NATUREZA DO ANEL:
    <rodízio de vencedores | dominador + fachada | especialista regional legítimo | indefinido>

    EVIDÊNCIAS:
    - <evidência citando o número que a sustenta>

    DILIGÊNCIAS PRIORITÁRIAS:
    - <o que o auditor deveria verificar primeiro>

    CLASSIFICAÇÃO DE RISCO: <🔴 ALTO | 🟡 MÉDIO | 🟢 BAIXO>
    """,
        ),
    ]
)


def build_supplier_graph(df, min_shared_units):
    """Constrói o grafo fornecedor↔fornecedor ligado por unidades gestoras compartilhadas.

    Retorna (igraph.Graph, lista de supplier_ids alinhada aos vértices) ou (None, None) se
    não houver arestas acima do limiar.
    """
    # Conjunto de fornecedores por unidade gestora.
    unit_groups = df.groupby(UNIT_COL)[SUPPLIER_ID_COL].apply(lambda s: sorted(set(s)))

    pair_counts = Counter()
    for suppliers in unit_groups:
        for a, b in combinations(suppliers, 2):
            pair_counts[(a, b)] += 1  # +1 unidade gestora compartilhada

    edges_weighted = [
        (a, b, w) for (a, b), w in pair_counts.items() if w >= min_shared_units
    ]
    if not edges_weighted:
        return None, None

    nodes = sorted({a for a, _, _ in edges_weighted} | {b for _, b, _ in edges_weighted})
    index = {s: i for i, s in enumerate(nodes)}

    G = ig.Graph(
        n=len(nodes),
        edges=[(index[a], index[b]) for a, b, _ in edges_weighted],
        directed=False,
    )
    G.es["weight"] = [w for _, _, w in edges_weighted]
    return G, nodes


def _mean(series):
    return pd.to_numeric(series, errors="coerce").mean()


def ring_metrics(ring_df, members, name_map):
    """Métricas quantitativas de um anel de fornecedores."""
    units_per_member = ring_df.groupby(UNIT_COL)[SUPPLIER_ID_COL].nunique()
    shared_units = int((units_per_member >= 2).sum())  # UGs com >=2 membros do anel

    member_counts = ring_df[SUPPLIER_ID_COL].value_counts()
    top_members = [
        f"{name_map.get(sid, sid)} ({int(cnt)} reg.)"
        for sid, cnt in member_counts.head(8).items()
    ]

    mean_unique = _mean(ring_df["unique"])
    return {
        "n_suppliers": len(members),
        "n_records": len(ring_df),
        "n_units": int(ring_df[UNIT_COL].nunique()),
        "shared_units": shared_units,
        "mean_unique": mean_unique,
        "mean_win": _mean(ring_df["win"]),
        "mean_risk": _mean(ring_df["Risk_Mean"]) if "Risk_Mean" in ring_df else np.nan,
        "fraud_records": int(pd.to_numeric(ring_df.get("fraude"), errors="coerce").fillna(0).sum())
        if "fraude" in ring_df else 0,
        "clusters_touched": int(ring_df["Cluster ID"].nunique()) if "Cluster ID" in ring_df else 0,
        "top_members": top_members,
        # Recorrência (UGs compartilhadas) ponderada pela rotação (1 - unique).
        "suspicion_score": round(shared_units * (1 - (mean_unique if pd.notna(mean_unique) else 0)), 3),
    }


def _fmt(v, d=3):
    return "N/D" if pd.isna(v) else f"{v:.{d}f}"


def format_ring_summary(ring_id, m):
    return "\n".join([
        f"ANEL {ring_id}: {m['n_suppliers']} fornecedores, {m['n_records']} registros vencedores",
        f"  Unidades gestoras cobertas: {m['n_units']} | compartilhadas por >=2 membros: {m['shared_units']}",
        f"  Proporção média de vencedores distintos (unique): {_fmt(m['mean_unique'])} "
        f"(quanto MENOR, mais rotação)",
        f"  win médio: {_fmt(m['mean_win'])} | risco GMM médio (Risk_Mean): {_fmt(m['mean_risk'])}",
        f"  Fraudes confirmadas no anel: {m['fraud_records']} | clusters semânticos tocados: {m['clusters_touched']}",
        "  Membros (por nº de registros): " + "; ".join(m["top_members"]),
    ])


def detect_rings_for_product(product, cfg, ring_chain):
    """Detecta e caracteriza anéis de fornecedores para um único produto."""
    paths = cfg["paths"]
    coll = cfg["collusion_network"]

    clustered_file = os.path.join(paths["processed"], f"{product}_clustered.csv")
    output_directory = os.path.join(paths["output"], product, "collusion_rings")
    rings_csv = os.path.join(paths["processed"], f"{product}_suspected_rings.csv")

    if not os.path.exists(clustered_file):
        print(f"[{product}] {clustered_file} não encontrado, pulando.")
        return

    os.makedirs(output_directory, exist_ok=True)

    df = pd.read_csv(clustered_file)
    df = df[df[SUPPLIER_ID_COL].notna()].copy()
    df[SUPPLIER_ID_COL] = df[SUPPLIER_ID_COL].astype(str)

    # Nome de exibição: o mais frequente por CNPJ (nomes têm variações).
    name_map = (
        df.groupby(SUPPLIER_ID_COL)[SUPPLIER_NAME_COL]
        .agg(lambda s: s.dropna().astype(str).mode().iat[0] if not s.dropna().empty else "")
        .to_dict()
    )

    min_shared_units = int(coll.get("min_shared_units", 2))
    min_ring_size = int(coll.get("min_ring_size", 3))
    top_rings_llm = int(coll.get("top_rings_llm", 10))

    G, nodes = build_supplier_graph(df, min_shared_units)
    if G is None:
        print(f"[{product}] nenhuma aresta com >= {min_shared_units} UGs compartilhadas, sem anéis.")
        return

    partition = leidenalg.find_partition(
        G, leidenalg.ModularityVertexPartition, weights="weight"
    )

    # Agrupa fornecedores por comunidade e mantém apenas anéis acima do tamanho mínimo.
    communities = {}
    for node_idx, comm in enumerate(partition.membership):
        communities.setdefault(comm, []).append(nodes[node_idx])

    rings = []
    for members in communities.values():
        if len(members) < min_ring_size:
            continue
        ring_df = df[df[SUPPLIER_ID_COL].isin(members)]
        rings.append((members, ring_metrics(ring_df, members, name_map)))

    if not rings:
        print(f"[{product}] nenhuma comunidade com >= {min_ring_size} fornecedores.")
        return

    # Ordena por suspeita decrescente e atribui Ring ID estável.
    rings.sort(key=lambda r: r[1]["suspicion_score"], reverse=True)

    records = []
    for ring_id, (members, m) in enumerate(rings):
        records.append({
            "Ring ID": ring_id,
            "Suspicion Score": m["suspicion_score"],
            "Fornecedores": m["n_suppliers"],
            "Registros": m["n_records"],
            "Unidades Gestoras": m["n_units"],
            "UGs Compartilhadas": m["shared_units"],
            "Unique Medio": round(m["mean_unique"], 4) if pd.notna(m["mean_unique"]) else None,
            "Win Medio": round(m["mean_win"], 4) if pd.notna(m["mean_win"]) else None,
            "Risk_Mean Medio": round(m["mean_risk"], 4) if pd.notna(m["mean_risk"]) else None,
            "Fraudes Confirmadas": m["fraud_records"],
            "Clusters Tocados": m["clusters_touched"],
            "Membros": "; ".join(name_map.get(s, s) for s in members),
        })

    pd.DataFrame(records).to_csv(rings_csv, index=False)
    print(f"[{product}] {len(rings)} anéis detectados → {rings_csv}")

    # Caracterização LLM dos anéis mais suspeitos.
    print(f"[{product}] caracterizando os {min(top_rings_llm, len(rings))} anéis mais suspeitos...")
    for ring_id, (members, m) in enumerate(rings[:top_rings_llm]):
        output_path = os.path.join(output_directory, f"ring_{ring_id}.txt")
        if os.path.exists(output_path):  # retomada
            continue

        raw_output = ring_chain.invoke({"ring_summary": format_ring_summary(ring_id, m)})
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw_output.content)
        print(f"[{product}] anel {ring_id} caracterizado.")


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
    ring_chain = RING_PROMPT | llm

    for product in cfg["products"]:
        print(f"=== Rede de conluio: {product} ===")
        detect_rings_for_product(product, cfg, ring_chain)


def convert_product(product, cfg):
    """Anexa a caracterização textual (.txt) ao CSV de anéis suspeitos do produto."""
    paths = cfg["paths"]

    rings_csv = os.path.join(paths["processed"], f"{product}_suspected_rings.csv")
    ring_dir = os.path.join(paths["output"], product, "collusion_rings")
    output_file = os.path.join(paths["processed"], f"{product}_suspected_rings_characterized.csv")

    if not os.path.exists(rings_csv):
        print(f"[{product}] {rings_csv} não encontrado, pulando.")
        return

    files = glob(os.path.join(ring_dir, "*.txt"))
    if not files:
        print(f"[{product}] nenhuma caracterização em {ring_dir}, pulando.")
        return

    records = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = clean_llm_text(f.read())
        records.append({"Ring ID": extract_cluster_id(file), "Caracterizacao": content})

    char_df = pd.DataFrame(records)
    rings_df = pd.read_csv(rings_csv)

    final_df = rings_df.merge(char_df, on="Ring ID", how="left")
    final_df.to_csv(output_file, index=False)
    print(f"[{product}] dataset de anéis caracterizados criado: {output_file}")


def json_converter():
    chdir_to_project_root()
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    for product in cfg["products"]:
        print(f"=== Conversão de anéis: {product} ===")
        convert_product(product, cfg)


if __name__ == "__main__":
    main()
    json_converter()
