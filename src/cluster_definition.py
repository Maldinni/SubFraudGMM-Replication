import os
import numpy as np
import pandas as pd
import sys
from glob import glob
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from utils.parsing import parse_args, load_config, extract_cluster_id, parse_cluster_definition
from utils.hypersphere import get_representative_articles, get_centroids
from utils.load_data import ensure_dirs, load_embedding_shards, align_to_df

def truncate(text, max_chars=4000):
    return text[:max_chars]

def main():
# Define prompt templates
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

    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    required_fields = cfg['definition']['required_fields']

    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=cfg['llm']['temperature']
    )

    cluster_directory = Path(cfg["paths"]["processed"]) / "separated clusters"
    cluster_files = glob(os.path.join(cluster_directory, "cluster_*.csv"))
    output_directory = f'{cfg["paths"]["output"]}'

    solicitacoes_file = Path(cfg["paths"]["processed"]) / "trator_esteira_final_merged_clustered.csv"

    solicitacoes_df = pd.read_csv(solicitacoes_file)

    # Garante que as métricas de risco vindas do SubFraudGMM (merge à esquerda,
    # podem conter NaN para registros sem correspondência) sejam numéricas.
    risk_columns = ["Risk_Mean", "Risk_Std", "Risk_Max", "Rank_Mean"]
    for col in risk_columns:
        if col in solicitacoes_df.columns:
            solicitacoes_df[col] = pd.to_numeric(solicitacoes_df[col], errors="coerce")

    def _fmt(series, decimals=None):
        """Formata uma coluna para texto, preservando legibilidade e tratando ausências."""
        s = series.round(decimals) if decimals is not None else series
        return s.astype(object).where(series.notna(), "N/D").astype(str)

    # O texto enviado ao auditor LLM precisa carregar os sinais quantitativos do
    # SubFraudGMM (Risk_*/Rank) além dos metadados, senão o modelo "raciocina" sem
    # acesso ao próprio indicador de risco que fundamenta o trabalho.
    solicitacoes_df["texto_cluster"] = (
        "Município: " + solicitacoes_df["Ente"].astype(str)
        + ". Empresa: " + solicitacoes_df["nomeParticipante"].astype(str)
        + ". Objeto: " + solicitacoes_df["Descrição Item Licitação"].astype(str)
        + ". Ano: " + solicitacoes_df["Ano"].astype(str)
        + ". Valor total cotado: " + solicitacoes_df["Valor Total Cotado Item"].astype(str)
        + ". Preço unitário: " + _fmt(solicitacoes_df["unit_price"], 2)
        + ". Participantes: " + solicitacoes_df["num_partic"].astype(str)
        + ". Venceu este item: " + _fmt(solicitacoes_df["auction_winner_flag"])
        + ". Taxa de vitória do fornecedor na unidade (win): " + _fmt(solicitacoes_df["win"], 3)
        + ". Duração do certame (dias): " + _fmt(solicitacoes_df["duration"])
        + ". Proporção de vencedores únicos na unidade (unique): " + _fmt(solicitacoes_df["unique"], 3)
        + ". Risco médio GMM (Risk_Mean): " + _fmt(solicitacoes_df["Risk_Mean"], 3)
        + ". Desvio do risco (Risk_Std): " + _fmt(solicitacoes_df["Risk_Std"], 3)
        + ". Risco máximo (Risk_Max): " + _fmt(solicitacoes_df["Risk_Max"], 3)
        + ". Ranking de risco (Rank_Mean): " + _fmt(solicitacoes_df["Rank_Mean"])
        + ". Cluster semântico: " + solicitacoes_df["Cluster ID"].astype(str)
    )

    print(solicitacoes_df.iloc[0].to_dict())

    cluster_csv_file = 'clusters_defined.csv'
    output_path = os.path.join(output_directory, cluster_csv_file)

    if os.path.exists(output_path):
        cluster_definitions_df = pd.read_csv(output_path)
        cluster_definitions = cluster_definitions_df.to_dict('records')
        processed_clusters = set(cluster_definitions_df['Cluster ID'].values.tolist())
    else:
        cluster_definitions = []
        processed_clusters = set()
    
    shard_directory = Path(cfg["paths"]["raw"]) / "shards_h5"
    files = glob(os.path.join(shard_directory, '*.h5'))

    embeddings, texts, ids = load_embedding_shards(files)

    print("Primeiros IDs do dataframe:")
    print(solicitacoes_df["ID"].head(10).tolist())

    print("Primeiros IDs dos embeddings:")
    print(ids[:10])

    print("Tipo df ID:", type(solicitacoes_df["ID"].iloc[0]))
    print("Tipo embedding ID:", type(ids[0]))

    aligned_embeddings, aligned_ids = align_to_df(
        embeddings,
        ids,
        solicitacoes_df
    )

    centroids = get_centroids(
    aligned_embeddings,
    solicitacoes_df['Cluster ID'].values
    )

    #sources = solicitacoes_df['source'].values
    definition_chain = DEFINITION_PROMPT | llm

    print("Iniciando processamento de clusters...")

    for cluster_file in cluster_files:

        label = int(Path(cluster_file).stem.split("_")[1])

        if label in processed_clusters:
            continue

        print(f"Processando cluster {label}...")

        cluster_df = pd.read_csv(cluster_file)

        number_of_articles = min(
            cfg['definition']['max_article_number'],
            len(cluster_df)
        )

        cluster_mask = solicitacoes_df['Cluster ID'] == label

        cluster_embeddings = aligned_embeddings[cluster_mask]

        cluster_texts = solicitacoes_df.loc[
            cluster_mask,
            "texto_cluster"
        ].values

        cluster_articles = get_representative_articles(
            centroids[label],
            cluster_embeddings,
            cluster_texts,
            number_of_articles
        )

        cluster_articles = [truncate(a) for a in cluster_articles]

        chain_input = {"articles": "\n\n".join(cluster_articles)}

        raw_output = definition_chain.invoke(chain_input)

        with open(f"{output_directory}/cluster_{label}.txt", "w", encoding="utf-8") as f:
            f.write(raw_output.content)

        print(f"Cluster {label} salvo.")

def json_converter():

    args = parse_args()
    cfg = load_config(args)

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    clusters_directory = f'{cfg["paths"]["output"]}'
    output_directory = f'{cfg["paths"]["processed"]}'

    cluster_json_file = 'clusters_licitacoes_defined.json'
    cluster_csv_file = 'clusters_licitacoes_defined.csv'
    dataset_file = Path(cfg["paths"]["processed"]) / "trator_esteira_final_merged_clustered.csv"

    shard_directory = Path(cfg["paths"]["raw"]) / "shards_h5"
    shards_files = glob(os.path.join(shard_directory, '*.h5'))

    # Arquivos de definição produzidos pelo auditor LLM (um .txt por cluster).
    definition_files = glob(os.path.join(clusters_directory, "cluster_*.txt"))

    df = pd.read_csv(dataset_file)

    # Considera apenas registros que receberam cluster (merge à esquerda e a
    # detecção de comunidades podem deixar Cluster ID nulo).
    df = df[df["Cluster ID"].notna()].copy()
    df["Cluster ID"] = df["Cluster ID"].astype(int)

    cluster_sizes = df.groupby("Cluster ID").size()

    cluster_records = []

    # Os embeddings vêm dos shards .h5 (não dos .txt de definição) e são
    # reordenados para casar com a ordem do dataframe.
    embeddings, texts, ids = load_embedding_shards(shards_files)

    aligned_embeddings, aligned_ids = align_to_df(
        embeddings,
        ids,
        df
    )

    labels = df['Cluster ID'].values
    centroids = get_centroids(aligned_embeddings, labels)

    # Mapeia cada rótulo de cluster para a linha correspondente na matriz de
    # centróides (get_centroids ordena por np.unique dos rótulos).
    unique_labels = np.unique(labels)
    label_to_row = {int(label): row for row, label in enumerate(unique_labels)}

    centroid_similarities = centroids.dot(centroids.T) - np.eye(centroids.shape[0])

    for file in definition_files:

        cluster_id = extract_cluster_id(file)

        if cluster_id not in label_to_row:
            print(f"Aviso: cluster {cluster_id} sem centróide correspondente, ignorando.")
            continue

        sections = parse_cluster_definition(file)

        row = label_to_row[cluster_id]
        most_similar_row = int(np.argmax(centroid_similarities[row]))
        most_similar_label = int(unique_labels[most_similar_row])
        similarity = float(centroid_similarities[row, most_similar_row])

        record = {
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
        }

        cluster_records.append(record)

    clusters_df = pd.DataFrame(cluster_records)

    clusters_df = clusters_df.sort_values("Cluster ID")

    clusters_df.to_csv(os.path.join(output_directory, cluster_csv_file), index=False)

    clusters_df.to_json(
        os.path.join(output_directory, cluster_json_file),
        orient="records",
        indent=2,
        force_ascii=False
    )

    print("Dataset criado com sucesso!")

if __name__ == '__main__':
    os.chdir('..')
    main()
    json_converter()