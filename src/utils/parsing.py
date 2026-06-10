import re
import argparse
import tomllib
from pathlib import Path
from typing import Any, Dict


def load_toml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    raw = p.read_bytes()
    with open(p, "rb") as f:
        return tomllib.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="News scraping prototype")

    p.add_argument("--directories", default="parameters/directories.toml")
    p.add_argument("--clustering", default="parameters/clustering.toml")
    p.add_argument("--initial_embedding", default="parameters/ingestion/initial_embedding.toml")
    p.add_argument("--semantic", default="parameters/analysis/semantic.toml")

    p.add_argument("--source", default=None, help="Rodar apenas uma fonte por nome.")
    p.add_argument("--max_articles_per_source", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")

    return p.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    dirs_file = load_toml(args.directories)
    initial_embedding_file = load_toml(args.initial_embedding)
    graph_construction_file = load_toml(args.clustering)
    cluster_definition_file = load_toml(args.semantic)
    trends_file = load_toml(args.semantic)

    cfg = {
        "paths": dirs_file["paths"],
        "initial_embedding": initial_embedding_file,
        "graph_construction": graph_construction_file["graph_construction"],
        "community_detection": graph_construction_file["community_detection"],
        "definition": cluster_definition_file["definition"],
        "llm": cluster_definition_file["llm"],
        "trends": trends_file["trends"],
    }

    return cfg

def extract_cluster_id(filename):

    # Usa apenas o nome do arquivo (ex.: 'cluster_3.txt') para evitar capturar
    # dígitos que apareçam no caminho do diretório.
    match = re.search(r"(\d+)", Path(filename).stem)
    return int(match.group(1))

def load_and_parse_cluster_file(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # keywords
    keywords_section = re.search(
        r"Palavras-chave:(.*?)Título:",
        text,
        re.S
    )

    keywords = []
    if keywords_section:
        keywords = re.findall(r"- (.*)", keywords_section.group(1))

    # title
    title_match = re.search(
        r"Título:\s*(.*?)\n",
        text
    )

    title = title_match.group(1).strip() if title_match else ""

    # description
    desc_match = re.search(
        r"Descrição:\s*(.*?)\n\s*Foco:",
        text,
        re.S
    )

    description = desc_match.group(1).strip() if desc_match else ""

    focus_match = re.search(
        r"Foco:\s*(.*)",
        text
    )

    focus = focus_match.group(1).strip() if focus_match else ""

    return keywords, title, description, focus

def parse_distinction_file(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    matches = re.findall(r"- (.*)", text)

    return "; ".join(matches)


def clean_llm_text(text):

    text = re.sub(r"#+", "", text)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"- ", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# Mapeia o número de cada seção do prompt do auditor (DEFINITION_PROMPT, [1]..[7])
# para um nome de campo legível usado no CSV/JSON consolidado.
CLUSTER_SECTION_FIELDS = {
    1: "Nome",
    2: "Tipo de Aquisicao",
    3: "Perfil dos Fornecedores",
    4: "Faixa de Valores",
    5: "Padroes Recorrentes",
    6: "Indicios de Risco",
    7: "Veredicto",
}

# Cabeçalho de seção: número entre colchetes, possivelmente precedido por
# marcações markdown (**, #, >) que o LLM costuma emitir.
_SECTION_HEADER = re.compile(r"^\s*[*#>\s]*\[(\d+)\]")


def parse_cluster_definition(filepath):
    """
    Parsing determinístico da saída textual do auditor LLM, que segue o formato
    de 7 seções numeradas ([1]..[7]) definido em DEFINITION_PROMPT.

    A linha de cabeçalho de cada seção é descartada; apenas o corpo é mantido.
    Retorna um dict {nome_do_campo: texto}, usando "Sem indícios detectados"
    como valor padrão para seções ausentes.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    raw_sections = {}
    current = None
    buffer = []
    for line in lines:
        match = _SECTION_HEADER.match(line)
        if match:
            if current is not None:
                raw_sections[current] = "".join(buffer)
            current = int(match.group(1))
            buffer = []
        elif current is not None:
            buffer.append(line)
    if current is not None:
        raw_sections[current] = "".join(buffer)

    parsed = {}
    for number, field in CLUSTER_SECTION_FIELDS.items():
        body = raw_sections.get(number, "").strip()
        parsed[field] = clean_llm_text(body) if body else "Sem indícios detectados"

    return parsed