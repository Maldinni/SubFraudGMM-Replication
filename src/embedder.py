import os
import time
import pickle
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from utils.parsing import parse_args, load_config
from utils.load_data import ensure_dirs, determine_output_filename, save_to_hdf5
from utils.checkpoints import load_processed_documents, save_processed_documents
from utils.embedding import unpack_embedding_parameters, save_embeddings, SentenceTransformerEmbeddings

from classes.data_types import Embeddings


def build_embedding_text(selection):
    """Monta o texto semântico de cada registro a ser embedado."""
    return (
        "Município: " + selection["Ente"].astype(str) + ". " +
        "Empresa: " + selection["nomeParticipante"].astype(str) + ". " +
        "Objeto: " + selection["Descrição Item Licitação"].astype(str) + ". " +
        "Ano: " + selection["Ano"].astype(str) + ". " +
        "Valor: " + selection["Valor Total Cotado Item"].astype(str) + ". " +
        "Participantes: " + selection["num_partic"].astype(str)
    )


def embed_product(product, cfg, embedding_model, sleep_time, items_per_shard):
    """Gera embeddings e shards .h5 para um único produto."""
    paths = cfg["paths"]

    cleaned_file = os.path.join(paths["processed"], f"{product}.csv")
    embedded_directory = os.path.join(paths["raw"], "embedded", product)
    shard_directory = os.path.join(paths["raw"], "shards_h5", product)
    checkpoint_path = os.path.join(paths["checkpoints"], f"{product}_processed_embedded.json")

    if not os.path.exists(cleaned_file):
        print(f"[{product}] arquivo {cleaned_file} não encontrado, pulando.")
        return

    os.makedirs(embedded_directory, exist_ok=True)
    os.makedirs(shard_directory, exist_ok=True)

    df = pd.read_csv(cleaned_file)

    # Remove registros já embedados (checkpoint por ID, permitindo retomada).
    processed = load_processed_documents(checkpoint_path)
    df = df[~df['ID'].isin(processed)]

    if len(df) == 0:
        print(f"[{product}] nada novo a embedar.")
    else:
        output_file, shard_id = determine_output_filename(embedded_directory, 'pkl')

        for start in tqdm(range(0, len(df), items_per_shard), desc=product):
            end = start + items_per_shard

            text_embeddings = Embeddings()

            selection = df.iloc[start:end].copy()
            selection["embedding_text"] = build_embedding_text(selection)
            texts = selection["embedding_text"].tolist()

            embedded_texts = embedding_model.embed_documents(texts)

            text_embeddings.ids = selection['ID'].tolist()
            text_embeddings.texts = texts
            text_embeddings.embeddings = embedded_texts

            save_embeddings(text_embeddings, output_file)

            # Atualiza o checkpoint pelos IDs efetivamente processados.
            processed.update(text_embeddings.ids)
            save_processed_documents(checkpoint_path, processed)

            shard_id += 1
            output_file = os.path.join(embedded_directory, f'shard_{shard_id:04d}.pkl')

            time.sleep(sleep_time)

    # Converte os pickles do produto em shards HDF5.
    embedded_directory = Path(embedded_directory)
    shard_directory = Path(shard_directory)

    for pkl_file in embedded_directory.glob("*.pkl"):
        print(f"[{product}] convertendo {pkl_file.name}...")

        with open(pkl_file, "rb") as f:
            obj = pickle.load(f)

        h5_filename = shard_directory / (pkl_file.stem + ".h5")

        news_list = []
        for i in range(len(obj.texts)):
            news_list.append({
                "id": obj.ids[i],
                "text": obj.texts[i],
                "embedding": obj.embeddings[i],
            })

        save_to_hdf5(news_list, str(h5_filename))


def main():
    os.chdir('..')  # Change to project root directory
    args = parse_args()
    cfg = load_config(args)

    model_name, sleep_time, batch_size, items_per_shard = unpack_embedding_parameters(
        cfg["initial_embedding"]
    )

    paths = cfg["paths"]
    ensure_dirs(paths["raw"], paths["processed"], paths["checkpoints"], paths["output"])

    # Modelo de embedding carregado uma única vez e reutilizado entre produtos.
    embedding_model = SentenceTransformerEmbeddings(
        model_name=model_name,
        batch_size=batch_size
    )

    for product in cfg["products"]:
        print(f"=== Embedding: {product} ===")
        embed_product(product, cfg, embedding_model, sleep_time, items_per_shard)

    print("Conversão finalizada.")


if __name__ == "__main__":
    main()
