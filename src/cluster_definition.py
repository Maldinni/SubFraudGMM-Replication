import os
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from utils.parsing import parse_args, load_config, extract_cluster_id, parse_cluster_definition
from utils.hypersphere import get_representative_articles, get_centroids
from utils.load_data import ensure_dirs, load_embedding_shards, align_to_df


DEFINITION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    Você é um auditor fiscal sênior com 20 anos de experiência em controle interno de licitações públicas brasileiras, especializado em detecção de fraudes, conluio entre fornecedores (cartel de licitação), direcionamento de edital e superfaturamento.

    Seu raciocínio segue a metodologia de auditoria baseada em risco do TCU (Tribunal de Contas da União). Você conhece profundamente:
    - A Lei 8.666/93, Lei 14.133/21 e Lei 10.520/02 (Pregão)
    - Indicadores quantitativos de risco: win rate anômalo, concentração de fornecedores, variação de preços, frequência de participação
    - Padrões comportamentais suspeitos: vencedor recorrente, participantes fictícios (bid-rigging), preços combinados, janelas de tempo curtas

    Ao analisar, você pensa em voz alta de forma estruturada antes de concluir. Nunca omite campos mesmo se a evidência for fraca — registre "Sem indícios detectados" nesses casos.
    """
        ),
        (
            "user",
            """
    ## TAREFA: ANÁLISE DE CLUSTER DE LICITAÇÕES

    Analise o seguinte conjunto de registros de licitações públicas brasileiras. Cada registro representa um item licitado com métricas de risco pré-calculadas.

    ### LEGENDA DOS CAMPOS
    - `Valor total cotado`: Valor total do item cotado (R$)
    - `Preço unitário (unit_price)`: Preço unitário praticado (R$)
    - `Participantes (num_partic)`: Número de participantes na disputa
    - `Venceu este item (auction_winner_flag)`: 1 = o fornecedor venceu este item; 0 = não
    - `Taxa de vitória (win)`: Proporção de vitórias do fornecedor na mesma unidade gestora (0–1; quanto maior, mais dominante)
    - `Duração do certame (duration)`: Dias entre abertura e homologação
    - `Vencedores únicos (unique)`: Proporção de vencedores distintos na unidade gestora (0–1; valores baixos indicam concentração)
    - `Risco médio/desvio/máximo (Risk_Mean / Risk_Std / Risk_Max)`: Índices de risco do SubFraudGMM (0–1; quanto maior, mais suspeito)
    - `Ranking de risco (Rank_Mean)`: Posição média de risco relativa ao universo total (menor = mais arriscado)
    - `Cluster semântico (Cluster ID)`: Agrupamento semântico do registro

    ### REGISTROS

    {articles}


    ---

    ### ANÁLISE REQUERIDA

    Raciocine passo a passo antes de preencher cada seção abaixo.

    **[1] NOME CURTO DO CLUSTER**
    Uma etiqueta de 2–5 palavras que identifique o grupo. Ex.: "Tratores de esteira — Santa Catarina"

    **[2] TIPO DE AQUISIÇÃO**
    Categoria predominante, modalidade licitatória inferida, e se as especificações técnicas parecem direcionadas a um fornecedor específico.

    **[3] PERFIL DOS FORNECEDORES**
    - Quem venceu? Há concentração em um único CNPJ/CPF?
    - Taxa de vitória (win rate) parece natural ou anômala?
    - Fornecedores são locais, regionais ou nacionais?

    **[4] FAIXA DE VALORES**
    - Preço mínimo, máximo e mediana unitária observada
    - Existe dispersão anormal de preços para o mesmo item?
    - Compare com o volume total contratado

    **[5] PADRÕES RECORRENTES**
    Identifique padrões temporais, geográficos, de participação ou de especificação que se repetem entre registros.

    **[6] INDÍCIOS DE RISCO OU FRAUDE**
    Para cada indício encontrado, cite o campo que o sustenta e classifique a severidade:
    🔴 ALTO | 🟡 MÉDIO | 🟢 BAIXO | ⚪ SEM INDÍCIO

    Tipos a verificar explicitamente:
    - Conluio/cartel (bid-rigging): poucos participantes + vencedor recorrente
    - Direcionamento de edital: especificação técnica restritiva
    - Superfaturamento: preço unitário muito acima da mediana
    - Fracionamento: múltiplas licitações de baixo valor para o mesmo item
    - Participante fictício: empresas que participam mas nunca vencem

    **[7] VEREDICTO CONSOLIDADO**
    Em 3–5 frases: qual é o nível de risco geral deste cluster? Quais diligências deveriam ser priorizadas?

    Responda APENAS com as seções numeradas acima, sem texto introdutório ou conclusivo fora delas.
    """
        )
    ]
)


def truncate(text, max_chars=4000):
    return text[:max_chars]


def build_texto_cluster(df):
    """
    Monta o texto por registro enviado ao auditor LLM, carregando metadados e os
    sinais quantitativos do SubFraudGMM (Risk_*/Rank). Sem isso, o modelo
    raciocinaria sem acesso ao próprio indicador de risco que fundamenta o trabalho.
    """
    # Garante que as métricas de risco (merge à esquerda pode deixar NaN) sejam numéricas.
    for col in ["Risk_Mean", "Risk_Std", "Risk_Max", "Rank_Mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _fmt(series, decimals=None):
        s = series.round(decimals) if decimals is not None else series
        return s.astype(object).where(series.notna(), "N/D").astype(str)

    return (
        "Município: " + df["Ente"].astype(str)
        + ". Empresa: " + df["nomeParticipante"].astype(str)
        + ". Objeto: " + df["Descrição Item Licitação"].astype(str)
        + ". Ano: " + df["Ano"].astype(str)
        + ". Valor total cotado: " + df["Valor Total Cotado Item"].astype(str)
        + ". Preço unitário: " + _fmt(df["unit_price"], 2)
        + ". Participantes: " + df["num_partic"].astype(str)
        + ". Venceu este item: " + _fmt(df["auction_winner_flag"])
        + ". Taxa de vitória do fornecedor na unidade (win): " + _fmt(df["win"], 3)
        + ". Duração do certame (dias): " + _fmt(df["duration"])
        + ". Proporção de vencedores únicos na unidade (unique): " + _fmt(df["unique"], 3)
        + ". Risco médio GMM (Risk_Mean): " + _fmt(df["Risk_Mean"], 3)
        + ". Desvio do risco (Risk_Std): " + _fmt(df["Risk_Std"], 3)
        + ". Risco máximo (Risk_Max): " + _fmt(df["Risk_Max"], 3)
        + ". Ranking de risco (Rank_Mean): " + _fmt(df["Rank_Mean"])
        + ". Cluster semântico: " + df["Cluster ID"].astype(str)
    )


def define_clusters_for_product(product, cfg, definition_chain):
    """Gera as definições textuais do auditor para os clusters de um produto."""
    paths = cfg["paths"]

    cluster_directory = os.path.join(paths["processed"], "separated clusters", product)
    cluster_files = glob(os.path.join(cluster_directory, "cluster_*.csv"))
    output_directory = os.path.join(paths["output"], product)
    solicitacoes_file = os.path.join(paths["processed"], f"{product}_clustered.csv")
    shard_directory = os.path.join(paths["raw"], "shards_h5", product)

    if not cluster_files:
        print(f"[{product}] nenhum cluster em {cluster_directory}, pulando.")
        return
    if not os.path.exists(solicitacoes_file):
        print(f"[{product}] {solicitacoes_file} não encontrado, pulando.")
        return

    os.makedirs(output_directory, exist_ok=True)

    solicitacoes_df = pd.read_csv(solicitacoes_file)

    # Considera apenas registros que receberam cluster.
    solicitacoes_df = solicitacoes_df[solicitacoes_df["Cluster ID"].notna()].copy()
    solicitacoes_df["Cluster ID"] = solicitacoes_df["Cluster ID"].astype(int)

    solicitacoes_df["texto_cluster"] = build_texto_cluster(solicitacoes_df)

    shards_files = glob(os.path.join(shard_directory, '*.h5'))
    embeddings, texts, ids = load_embedding_shards(shards_files)

    aligned_embeddings, aligned_ids = align_to_df(embeddings, ids, solicitacoes_df)

    labels = solicitacoes_df['Cluster ID'].values
    centroids = get_centroids(aligned_embeddings, labels)

    unique_labels = np.unique(labels)
    label_to_row = {int(label): row for row, label in enumerate(unique_labels)}

    print(f"[{product}] iniciando definição de {len(cluster_files)} clusters...")

    for cluster_file in cluster_files:

        label = int(Path(cluster_file).stem.split("_")[1])

        output_path = os.path.join(output_directory, f"cluster_{label}.txt")

        # Retomada: pula clusters já definidos.
        if os.path.exists(output_path):
            continue

        if label not in label_to_row:
            print(f"[{product}] cluster {label} sem centróide correspondente, ignorando.")
            continue

        cluster_df = pd.read_csv(cluster_file)
        number_of_articles = min(cfg['definition']['max_article_number'], len(cluster_df))

        cluster_mask = solicitacoes_df['Cluster ID'] == label
        cluster_embeddings = aligned_embeddings[cluster_mask]
        cluster_texts = solicitacoes_df.loc[cluster_mask, "texto_cluster"].values

        cluster_articles = get_representative_articles(
            centroids[label_to_row[label]],
            cluster_embeddings,
            cluster_texts,
            number_of_articles
        )

        cluster_articles = [truncate(a) for a in cluster_articles]
        chain_input = {"articles": "\n\n".join(cluster_articles)}

        raw_output = definition_chain.invoke(chain_input)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(raw_output.content)

        print(f"[{product}] cluster {label} salvo.")


def main():
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    llm = ChatOllama(
        model=cfg['llm'].get('model_name', 'qwen2.5:7b'),
        temperature=cfg['llm']['temperature']
    )
    definition_chain = DEFINITION_PROMPT | llm

    for product in cfg["products"]:
        print(f"=== Definição de clusters: {product} ===")
        define_clusters_for_product(product, cfg, definition_chain)


def convert_product(product, cfg):
    """Converte as definições textuais (.txt) de um produto em CSV/JSON estruturados."""
    paths = cfg["paths"]

    clusters_directory = os.path.join(paths["output"], product)
    dataset_file = os.path.join(paths["processed"], f"{product}_clustered.csv")
    shard_directory = os.path.join(paths["raw"], "shards_h5", product)

    cluster_csv_file = os.path.join(paths["processed"], f"{product}_clusters_defined.csv")
    cluster_json_file = os.path.join(paths["processed"], f"{product}_clusters_defined.json")

    definition_files = glob(os.path.join(clusters_directory, "cluster_*.txt"))

    if not definition_files:
        print(f"[{product}] nenhuma definição .txt em {clusters_directory}, pulando.")
        return
    if not os.path.exists(dataset_file):
        print(f"[{product}] {dataset_file} não encontrado, pulando.")
        return

    df = pd.read_csv(dataset_file)
    df = df[df["Cluster ID"].notna()].copy()
    df["Cluster ID"] = df["Cluster ID"].astype(int)

    cluster_sizes = df.groupby("Cluster ID").size()

    shards_files = glob(os.path.join(shard_directory, '*.h5'))
    embeddings, texts, ids = load_embedding_shards(shards_files)

    aligned_embeddings, aligned_ids = align_to_df(embeddings, ids, df)

    labels = df['Cluster ID'].values
    centroids = get_centroids(aligned_embeddings, labels)

    unique_labels = np.unique(labels)
    label_to_row = {int(label): row for row, label in enumerate(unique_labels)}

    centroid_similarities = centroids.dot(centroids.T) - np.eye(centroids.shape[0])

    cluster_records = []

    for file in definition_files:

        cluster_id = extract_cluster_id(file)

        if cluster_id not in label_to_row:
            print(f"[{product}] cluster {cluster_id} sem centróide correspondente, ignorando.")
            continue

        sections = parse_cluster_definition(file)

        row = label_to_row[cluster_id]
        most_similar_row = int(np.argmax(centroid_similarities[row]))
        most_similar_label = int(unique_labels[most_similar_row])
        similarity = float(centroid_similarities[row, most_similar_row])

        cluster_records.append({
            "Cluster ID": cluster_id,
            "Tamanho": int(cluster_sizes.get(cluster_id, 0)),
            "Nome": sections["Nome"],
            "Tipo de Aquisicao": sections["Tipo de Aquisicao"],
            "Perfil dos Fornecedores": sections["Perfil dos Fornecedores"],
            "Faixa de Valores": sections["Faixa de Valores"],
            "Padroes Recorrentes": sections["Padroes Recorrentes"],
            "Indicios de Risco": sections["Indicios de Risco"],
            "Veredicto": sections["Veredicto"],
            "Cluster Mais Similar": most_similar_label,
            "Similaridade": similarity,
        })

    clusters_df = pd.DataFrame(cluster_records).sort_values("Cluster ID")

    clusters_df.to_csv(cluster_csv_file, index=False)
    clusters_df.to_json(cluster_json_file, orient="records", indent=2, force_ascii=False)

    print(f"[{product}] dataset de definições criado com sucesso!")


def json_converter():
    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    for product in cfg["products"]:
        print(f"=== Conversão JSON: {product} ===")
        convert_product(product, cfg)


if __name__ == '__main__':
    os.chdir('..')
    main()
    json_converter()
