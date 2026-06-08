import pandas as pd

def calc_mean(df):
    """
    Consolida múltiplas observações do GMM em um único registro por ID.
    """

    df_mean = (
        df.groupby(["ID", "Produto"])
        .agg({
            "Risk Indicator": ["mean", "std", "max"],
            "Rank": "mean"
        })
        .reset_index()
    )

    df_mean.columns = [
        "ID",
        "Produto",
        "Risk_Mean",
        "Risk_Std",
        "Risk_Max",
        "Rank_Mean"
    ]

    df_mean["Risk_Std"] = df_mean["Risk_Std"].fillna(0)

    # Converte Rank_Mean para inteiro
    df_mean["Rank_Mean"] = df_mean["Rank_Mean"].round().astype(int)

    return df_mean