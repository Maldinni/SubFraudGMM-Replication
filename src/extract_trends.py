"""
Tendências temporais por cluster — etapa opcional após cluster_definition/distinction.

Para cada cluster semântico, divide os registros por ANO em dois períodos (antigo vs.
recente) e extrai, de forma determinística, a movimentação dos ATORES (fornecedores em
alta/baixa) e a tendência dos sinais de risco (preço unitário, win, unique, num_partic).
Esses agregados são entregues ao auditor LLM, que narra a evolução e classifica o risco.

Por que isso importa no contexto atual (vindo de um sistema anterior de chamados, onde
rastreava motivos de recusa ao longo do tempo): num grupo de licitações quase idênticas,
um fornecedor que SOBE até dominar indica captura de mercado / consolidação de cartel; um
que SOME sugere rodízio. Preço unitário subindo é sobrepreço crescente; num_partic caindo
é a competição secando. A leitura temporal complementa o retrato estático do auditor.

Roda DEPOIS de cluster_definition e itera por produto, como o restante do pipeline.
"""

import os
import pandas as pd
from glob import glob

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


TRENDS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    Você é um auditor fiscal sênior em licitações públicas brasileiras, analisando a EVOLUÇÃO
    TEMPORAL de um grupo de licitações de objeto semelhante. Você recebe agregados já calculados
    de dois períodos (antigo e recente) e a movimentação dos fornecedores entre eles.

    Interprete os números sob a ótica de risco de fraude:
    - Fornecedor que sobe até dominar o grupo → captura de mercado / consolidação de cartel.
    - Fornecedor que desaparece → possível rodízio de vencedores.
    - Preço unitário médio subindo acima da inflação → sobrepreço crescente.
    - win/unique piorando → concentração; num_partic caindo → competição secando (cartel).

    Use SOMENTE os números fornecidos. Não invente fornecedores nem valores.
    """,
        ),
        (
            "user",
            """
    ## TAREFA: TENDÊNCIAS TEMPORAIS DO CLUSTER

    Cluster: {title}
    Corte temporal: período ANTIGO < {date} <= período RECENTE

    ### AGREGADOS POR PERÍODO
    {old_summary}

    {recent_summary}

    ### MOVIMENTAÇÃO DE ATORES (variação de participação recente − antiga)
    {actor_movement}

    ### TENDÊNCIA DOS SINAIS (recente − antigo)
    {signal_trends}

    ---

    Responda EXATAMENTE no formato abaixo, sem texto fora dele.

    ATORES EM ALTA:
    - <fornecedor e o que o sustenta>

    ATORES EM BAIXA:
    - <fornecedor e o que o sustenta>

    TENDÊNCIAS DOS SINAIS:
    - <preço/win/unique/participantes — o que mudou>

    LEITURA DE RISCO:
    <1–2 frases interpretando a evolução; classifique 🔴 ALTO | 🟡 MÉDIO | 🟢 BAIXO>
    """,
        ),
    ]
)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def add_year_column(df):
    """Deriva um ano inteiro confiável de ``Ano`` (com fallback para ``Data Homologacao``)."""
    year = _num(df.get("Ano"))
    if "Data Homologacao" in df.columns:
        dh_year = pd.to_datetime(df["Data Homologacao"], errors="coerce", utc=True).dt.year
        year = year.where(year.between(2000, 2100), dh_year)
    df = df.copy()
    df["year"] = year
    df = df[df["year"].between(2000, 2100)].copy()
    df["year"] = df["year"].astype(int)
    return df


def _fmt(value, decimals=2):
    return "N/D" if pd.isna(value) else f"{value:.{decimals}f}"


def summarize_period(df_p, top_actors):
    """Resumo determinístico de um período: médias dos sinais e principais fornecedores."""
    wins = _num(df_p["auction_winner_flag"])
    actors = (
        df_p.assign(_win=wins)
        .groupby("nomeParticipante")
        .agg(participacoes=("nomeParticipante", "size"), vitorias=("_win", "sum"))
        .sort_values("participacoes", ascending=False)
        .head(top_actors)
    )
    return {
        "n": len(df_p),
        "price": _num(df_p["unit_price"]).mean(),
        "win": _num(df_p["win"]).mean(),
        "unique": _num(df_p["unique"]).mean(),
        "partic": _num(df_p["num_partic"]).mean(),
        "actors": actors,
    }


def format_period(name, s):
    lines = [
        f"{name}: {s['n']} registros",
        f"  Preço unitário médio: {_fmt(s['price'])} | win médio: {_fmt(s['win'], 3)} | "
        f"unique médio: {_fmt(s['unique'], 3)} | participantes médio: {_fmt(s['partic'], 1)}",
        "  Principais fornecedores (participações/vitórias):",
    ]
    if s["actors"].empty:
        lines.append("    - (nenhum)")
    for nome, row in s["actors"].iterrows():
        lines.append(f"    - {nome}: {int(row['participacoes'])}/{int(row['vitorias'])}")
    return "\n".join(lines)


def format_actor_movement(older, recent, top_actors):
    """Variação de participação (share) de cada fornecedor entre os períodos."""
    oc = older["nomeParticipante"].value_counts()
    rc = recent["nomeParticipante"].value_counts()
    o_share = oc / oc.sum() if oc.sum() else oc
    r_share = rc / rc.sum() if rc.sum() else rc

    actors = oc.index.union(rc.index)
    delta = (
        r_share.reindex(actors, fill_value=0) - o_share.reindex(actors, fill_value=0)
    ).sort_values(ascending=False)

    def _tag(nome):
        if nome not in oc.index:
            return " (novo)"
        if nome not in rc.index:
            return " (saiu)"
        return ""

    alta = delta[delta > 0].head(top_actors)
    baixa = delta[delta < 0].tail(top_actors).sort_values()

    lines = ["EM ALTA:"]
    lines += [f"  - {nome}{_tag(nome)}: +{d:.2f} share" for nome, d in alta.items()] or ["  - (nenhum)"]
    lines.append("EM BAIXA:")
    lines += [f"  - {nome}{_tag(nome)}: {d:.2f} share" for nome, d in baixa.items()] or ["  - (nenhum)"]
    return "\n".join(lines)


def format_signal_trends(older_s, recent_s):
    def _delta(a, b):
        if pd.isna(a) or pd.isna(b):
            return "N/D"
        diff = b - a
        return f"{diff:+.3f}"

    return (
        f"  Preço unitário: {_delta(older_s['price'], recent_s['price'])} | "
        f"win: {_delta(older_s['win'], recent_s['win'])} | "
        f"unique: {_delta(older_s['unique'], recent_s['unique'])} | "
        f"participantes: {_delta(older_s['partic'], recent_s['partic'])}"
    )


def extract_trends_for_product(product, cfg, trends_chain):
    """Extrai tendências temporais por cluster para um único produto."""
    paths = cfg["paths"]

    clustered_file = os.path.join(paths["processed"], f"{product}_clustered.csv")
    # Usa o CSV mais "rico" disponível só para obter o nome do cluster.
    name_file = os.path.join(paths["processed"], f"{product}_clusters_distinguished.csv")
    if not os.path.exists(name_file):
        name_file = os.path.join(paths["processed"], f"{product}_clusters_defined.csv")
    output_directory = os.path.join(paths["output"], product, "extract_trends")

    if not os.path.exists(clustered_file):
        print(f"[{product}] {clustered_file} não encontrado, pulando.")
        return

    os.makedirs(output_directory, exist_ok=True)

    df = pd.read_csv(clustered_file)
    df = df[df["Cluster ID"].notna()].copy()
    df["Cluster ID"] = df["Cluster ID"].astype(int)
    df = add_year_column(df)

    cluster_names = {}
    if os.path.exists(name_file):
        name_df = pd.read_csv(name_file)
        if "Nome" in name_df.columns:
            cluster_names = dict(zip(name_df["Cluster ID"].astype(int), name_df["Nome"]))

    older_cutoff = int(cfg["trends"]["older_cutoff"])
    recent_cutoff = int(cfg["trends"]["recent_cutoff"])
    top_actors = int(cfg["trends"]["top_actors"])

    print(f"[{product}] extraindo tendências de {df['Cluster ID'].nunique()} clusters...")

    for label in sorted(df["Cluster ID"].unique()):
        label = int(label)
        output_path = os.path.join(output_directory, f"cluster_{label}_trends.txt")
        if os.path.exists(output_path):  # retomada
            continue

        cluster_df = df[df["Cluster ID"] == label]
        older = cluster_df[(cluster_df["year"] >= older_cutoff) & (cluster_df["year"] < recent_cutoff)]
        recent = cluster_df[cluster_df["year"] >= recent_cutoff]

        # Sem os dois períodos não há tendência a comparar.
        if older.empty or recent.empty:
            continue

        older_s = summarize_period(older, top_actors)
        recent_s = summarize_period(recent, top_actors)

        chain_input = {
            "title": cluster_names.get(label, f"Cluster {label}"),
            "date": recent_cutoff,
            "old_summary": format_period(f"PERÍODO ANTIGO (anos {older_cutoff}–{recent_cutoff - 1})", older_s),
            "recent_summary": format_period(f"PERÍODO RECENTE (anos >= {recent_cutoff})", recent_s),
            "actor_movement": format_actor_movement(older, recent, top_actors),
            "signal_trends": format_signal_trends(older_s, recent_s),
        }

        raw_output = trends_chain.invoke(chain_input)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw_output.content)

        print(f"[{product}] cluster {label} salvo.")


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
    trends_chain = TRENDS_PROMPT | llm

    for product in cfg["products"]:
        print(f"=== Tendências temporais: {product} ===")
        extract_trends_for_product(product, cfg, trends_chain)


def convert_product(product, cfg):
    """Anexa as tendências textuais (.txt) ao CSV de clusters do produto."""
    paths = cfg["paths"]

    # Encadeia sobre o CSV mais completo disponível (distinguished > defined).
    base_file = os.path.join(paths["processed"], f"{product}_clusters_distinguished.csv")
    if not os.path.exists(base_file):
        base_file = os.path.join(paths["processed"], f"{product}_clusters_defined.csv")

    trends_dir = os.path.join(paths["output"], product, "extract_trends")
    output_file = os.path.join(paths["processed"], f"{product}_clusters_trends.csv")

    if not os.path.exists(base_file):
        print(f"[{product}] {base_file} não encontrado, pulando.")
        return

    files = glob(os.path.join(trends_dir, "*.txt"))
    if not files:
        print(f"[{product}] nenhuma tendência em {trends_dir}, pulando.")
        return

    records = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            content = clean_llm_text(f.read())
        records.append({"Cluster ID": extract_cluster_id(file), "Tendencias": content})

    trends_df = pd.DataFrame(records)
    cluster_df = pd.read_csv(base_file)

    final_df = cluster_df.merge(trends_df, on="Cluster ID", how="left")
    final_df.to_csv(output_file, index=False)

    print(f"[{product}] dataset com tendências criado: {output_file}")


def json_converter():
    chdir_to_project_root()
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    for product in cfg["products"]:
        print(f"=== Conversão de tendências: {product} ===")
        convert_product(product, cfg)


if __name__ == "__main__":
    main()
    json_converter()
