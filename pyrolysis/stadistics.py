import pandas as pd
import numpy as np
import os 


def analysis_df(file='default.csv', file_out='default.xlsx'):

    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file)

    df1 = pd.read_csv(output_path, header=[0, 1], index_col=0)
    desired_cols_lvl0 = [
        "Element", "P_moisture", "Ash", "U_carbon", "U_h", "U_o", "U_n", "U_s",
        "Treaction", "carbonconversion", "Tyre", "Diesel", "LFO", "Carbon", "Metals",
        "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"
    ]

    desired_cols_lvl1 = [
        "Feature",
        "Value P moisture [wt%]",
        "Value ash [wt%]",
        "Value c [wt%]",
        "Value h [wt%]",
        "Value o [wt%]",
        "Value n [wt%]",
        "Value s [wt%]",
        "Treaction [K]",
        "Conversion val [%wt]",
        "Tyre price [USD/kg]",
        "Product diesel price [USD/kg]",
        "Product lfo price [USD/kg]",
        "Product carbon price [USD/kg]",
        "Product metals price [USD/kg]",
        "FEDI [damage*meter]",
        "MSP diesel [USD/kg]",
        "MSP LFO [USD/kg]",
        "MSP carbon activated [USD/kg]",
        "MSP metals [USD/kg]",
        "TCI [10^6 * USD]",
        "NPV [10^6 * USD]",
        "IRR [%]",
        "GWP energy [kg-CO2e/kg]",
        "GWP revenue [kg-CO2e/kg]",
        "Carbon yield [%wt]",
        "Real conversion [conversion]"
    ]

    cols_to_use = []
    for col in df1.columns:
        lvl0, lvl1 = col
        if (lvl0 in desired_cols_lvl0) or (lvl1 in desired_cols_lvl1):
            cols_to_use.append(col)

    df_clean = df1[cols_to_use].copy()

    # Aplana multiindex
    df_clean.columns = [f"{a}__{b}" for a, b in df_clean.columns]

    # Buscar columnas IRR y FEDI dentro de df_clean
    IRR_col = [c for c in df_clean.columns if "IRR" in c][0]
    FEDI_col = [c for c in df_clean.columns if "FEDI" in c][0]

    df_final = df_clean.copy()
    df_final["IRR"] = df_clean[IRR_col]
    df_final["FEDI"] = df_clean[FEDI_col]

    # Clasificación por cuantiles 
    p25_IRR, p75_IRR = np.percentile(df_final["IRR"], [25, 75])
    p25_FEDI, p75_FEDI = np.percentile(df_final["FEDI"], [25, 75])

    def classify(row):
        if row["IRR"] > p75_IRR and row["FEDI"] < p25_FEDI:
            return "A_high_IRR_low_risk"
        elif row["IRR"] > p75_IRR and row["FEDI"] > p75_FEDI:
            return "B_high_IRR_high_risk"
        elif row["IRR"] < p25_IRR and row["FEDI"] < p25_FEDI:
            return "C_low_IRR_low_risk"
        else:
            return "D_low_IRR_high_risk"

    df_final["class"] = df_final.apply(classify, axis=1)

    # Estadísticas globales
    global_stats_df = df_final.describe()

    # IRR y FEDI por clase
    irr_fedi_by_class_df = df_final.groupby("class").agg(
        IRR_mean=("IRR", "mean"),
        IRR_std=("IRR", "std"),
        FEDI_mean=("FEDI", "mean"),
        FEDI_std=("FEDI", "std"),
        count=("class", "count"),
    )

    # Variables por clase: medias y desviaciones estándar  ( DATOS MAS IMPORTANTES)
    feature_cols = [c for c in df_final.columns if c not in ["IRR", "FEDI", "class"]]
    vars_mean_by_class_df = df_final.groupby("class")[feature_cols].mean()
    vars_std_by_class_df = df_final.groupby("class")[feature_cols].std()

    # Diferencia A – D (medias y std) ( DATOS MAS IMPORTANTES)
    diff_AD_df = pd.DataFrame()
    if "A_high_IRR_low_risk" in vars_mean_by_class_df.index and \
       "D_low_IRR_high_risk" in vars_mean_by_class_df.index:

        # Diferencia de medias
        diff_AD_mean = (
            vars_mean_by_class_df.loc["A_high_IRR_low_risk"]
            - vars_mean_by_class_df.loc["D_low_IRR_high_risk"]
        )

        diff_AD_df = diff_AD_mean.sort_values(ascending=False).to_frame("A_minus_D_mean")

        # Diferencia de desviaciones estándar entre A y D (ver si sirve de algo)
        if "A_high_IRR_low_risk" in vars_std_by_class_df.index and \
           "D_low_IRR_high_risk" in vars_std_by_class_df.index:
            diff_AD_std = (
                vars_std_by_class_df.loc["A_high_IRR_low_risk"]
                - vars_std_by_class_df.loc["D_low_IRR_high_risk"]
            )
            
            diff_AD_df["A_minus_D_std"] = diff_AD_std[diff_AD_df.index]

        diff_AD_df.index.name = "variable"

    output_path2 = os.path.join(output_folder, file_out)
    with pd.ExcelWriter(output_path2) as writer:
        global_stats_df.to_excel(writer, sheet_name="global_stats")
        irr_fedi_by_class_df.to_excel(writer, sheet_name="IRR_FEDI_por_clase")
        vars_mean_by_class_df.to_excel(writer, sheet_name="vars_mean_por_clase")
        vars_std_by_class_df.to_excel(writer, sheet_name="vars_std_por_clase")
        diff_AD_df.to_excel(writer, sheet_name="A_menos_D")
