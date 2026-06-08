from processing.normalization import normalize_columns
from processing.anonymization import anonimizar_cpf
from processing.mean import calc_mean
from cleaning.clean_data import clean_dataset
#def build_dataset():
#    df_original = load_all_controles()
#    df = normalize_columns(df_original)
#
#    return df

def build_dataset(df):
    df = normalize_columns(df)
    df_final = clean_dataset(df)

    return df_final

def build_dataset_anonymized(dataset):
    df = dataset

    print(df.columns.tolist())

    df["cpf_hash"] = df["cpf/cnpj_participante_cotacao"].apply(anonimizar_cpf)
    df = df.drop(columns=["cpf/cnpj_participante_cotacao"])

    #Para limpar o dataset antes de retornar o final
    df_final = clean_dataset(df)

    return df_final

def merge_dataset(df_intermediary, df_final):

    df_mean = calc_mean(df_final)

    merged_df = df_intermediary.merge(
        df_mean,
        on="ID",
        how="left"
    )

    return merged_df

#def build_db_dataset_for_llm(dataset):
    df = dataset

    #Para limpar o dataset antes de retornar o final
    df_final = clean_first_access_llm_dataset(df)

    return df_final