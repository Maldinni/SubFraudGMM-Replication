from processing.mean import calc_mean


def merge_dataset(df_intermediary, df_final):

    df_mean = calc_mean(df_final)

    merged_df = df_intermediary.merge(
        df_mean,
        on="ID",
        how="left"
    )

    return merged_df
