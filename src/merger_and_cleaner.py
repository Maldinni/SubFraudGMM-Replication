from pathlib import Path

from config.settings import PROCESSED_DIR, INTERMEDIARY_ESC_CSV_INPUT_FILE, INTERMEDIARY_MOT_CSV_INPUT_FILE, INTERMEDIARY_ROL_CSV_INPUT_FILE, INTERMEDIARY_TRA_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE
from utils.load_data import load_csv
from pipeline.build_dataset import build_dataset_anonymized, build_dataset, merge_dataset

def cleaner_anonymizer_intermediary(file_path):
    print(file_path)
    df = load_csv(file_path)
    df_normalized = build_dataset(df)
    print(df_normalized)
    df_anonymized = build_dataset_anonymized(df_normalized)

    df = df_anonymized

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{Path(file_path).stem}_limpo{Path(file_path).suffix}", index=False)

    print("Dataset limpo com sucesso!")

def merger(file_path_intermediary, file_path_final):
    df_intermediary = load_csv(file_path_intermediary)
    df_final = load_csv(file_path_final)
    merged_df = merge_dataset(df_intermediary, df_final)

    df = merged_df

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / f"{Path(file_path_intermediary).stem}_merged{Path(file_path_intermediary).suffix}", index=False)

    print("Dataset incorporado com sucesso!")

if __name__ == "__main__":
    cleaner_anonymizer_intermediary(INTERMEDIARY_ESC_CSV_INPUT_FILE)
    cleaner_anonymizer_intermediary(INTERMEDIARY_MOT_CSV_INPUT_FILE)
    cleaner_anonymizer_intermediary(INTERMEDIARY_ROL_CSV_INPUT_FILE)
    cleaner_anonymizer_intermediary(INTERMEDIARY_TRA_CSV_INPUT_FILE)
    merger(INTERMEDIARY_ESC_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE)
    merger(INTERMEDIARY_MOT_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE)
    merger(INTERMEDIARY_ROL_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE)
    merger(INTERMEDIARY_TRA_CSV_INPUT_FILE, FINAL_GMM_CSV_INPUT_FILE)