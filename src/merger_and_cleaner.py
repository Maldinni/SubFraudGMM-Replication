"""
Mescla os dados intermediários (já pré-processados na notebook 01) com o
ranking de risco final do SubFraudGMM, gerando os arquivos `*_merged.csv`
consumidos pela camada NLP (embedding → grafo → comunidades → auditor LLM).

NOTA SOBRE ANONIMIZAÇÃO:
Optou-se por NÃO anonimizar os identificadores (CPF/CNPJ) nem os nomes de
participantes. Os dados provêm da Operação Patrola (investigação da Polícia
Federal já tornada pública) e do sistema eSfinge do TCE-SP, sendo informações
de licitação pública de acesso aberto. Manter os identificadores em claro é
necessário para a análise de conluio entre fornecedores (detecção de cartel
e vencedor recorrente), que perderia sentido sob hashing. Ver justificativa
ética detalhada no TCC.
"""

from pathlib import Path

from config.settings import (
    PROCESSED_DIR,
    INTERMEDIARY_ESC_CSV_INPUT_FILE,
    INTERMEDIARY_MOT_CSV_INPUT_FILE,
    INTERMEDIARY_ROL_CSV_INPUT_FILE,
    INTERMEDIARY_TRA_CSV_INPUT_FILE,
    FINAL_GMM_CSV_INPUT_FILE,
)
from utils.load_data import load_csv
from pipeline.build_dataset import merge_dataset


def merger(file_path_intermediary, file_path_final):
    df_intermediary = load_csv(file_path_intermediary)
    df_final = load_csv(file_path_final)
    merged_df = merge_dataset(df_intermediary, df_final)

    df = merged_df

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{Path(file_path_intermediary).stem}_merged{Path(file_path_intermediary).suffix}", index=False)

    print("Dataset incorporado com sucesso!")


if __name__ == "__main__":
    merger(INTERMEDIARY_ESC_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE)
    merger(INTERMEDIARY_MOT_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE)
    merger(INTERMEDIARY_ROL_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE)
    merger(INTERMEDIARY_TRA_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE)
