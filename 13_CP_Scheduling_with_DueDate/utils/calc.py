import pandas as pd


# "Lateness" (Tardiness und Earliness) ------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
def compute_tardiness_earliness_ideal_ratios(df_plan_last_ops_list):
    """
    Berechnet für jeden Tag den Anteil an:
    - Tardiness > 0
    - Earliness > 0
    - Ideal (T=0 & E=0)

    Gibt NaN zurück, wenn ein DataFrame leer ist.
    """
    tardiness_ratio_per_day = []
    earliness_ratio_per_day = []
    ideal_ratio_per_day = []

    for df in df_plan_last_ops_list:
        tardiness_ratio = (df["Tardiness"] > 0).mean()
        earliness_ratio = (df["Earliness"] > 0).mean()
        ideal_ratio = ((df["Tardiness"] == 0) & (df["Earliness"] == 0)).mean()

        tardiness_ratio_per_day.append(tardiness_ratio)
        earliness_ratio_per_day.append(earliness_ratio)
        ideal_ratio_per_day.append(ideal_ratio)

    return tardiness_ratio_per_day, earliness_ratio_per_day, ideal_ratio_per_day

def compute_mean_tardiness_earliness(df_plan_last_ops_list):
    mean_tardiness_per_day = []
    mean_earliness_per_day = []

    for df in df_plan_last_ops_list:
        mean_tardiness = df["Tardiness"].mean()
        mean_earliness = df["Earliness"].mean()

        mean_tardiness_per_day.append(mean_tardiness)
        mean_earliness_per_day.append(mean_earliness)

    return mean_tardiness_per_day, mean_earliness_per_day


def compute_nonzero_mean_tardiness_earliness(df_plan_last_ops_list):
    mean_tardiness_per_day = []
    mean_earliness_per_day = []

    for df in df_plan_last_ops_list:
        # Nur Tardiness-Werte > 0 berücksichtigen
        tardiness_values = df["Tardiness"][df["Tardiness"] > 0]
        if not tardiness_values.empty:
            mean_tardiness = tardiness_values.mean()
        else:
            mean_tardiness = 0.0
        mean_tardiness_per_day.append(mean_tardiness)

        # Nur Earliness-Werte > 0 berücksichtigen
        earliness_values = df["Earliness"][df["Earliness"] > 0]
        if not earliness_values.empty:
            mean_earliness = earliness_values.mean()
        else:
            mean_earliness = 0.0
        mean_earliness_per_day.append(mean_earliness)

    return mean_tardiness_per_day, mean_earliness_per_day

# Deviation ---------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def compute_daily_starttime_deviations(plan_list, job_col="Job", op_col="Operation", start_col="Start", method="sum"):
    """
    Berechnet die tägliche Startzeit-Abweichung zwischen aufeinanderfolgenden Plänen.

    - Tag 0: Deviation = 0
    - Ab Tag 1: Vergleich mit jeweils vorherigem Tag

    Parameter:
    - plan_list: Liste von DataFrames mit Spalten für Job, Operation und Startzeit
    - method: "sum" für Gesamtabweichung, "mean" für durchschnittliche Abweichung pro Operation

    Rückgabe:
    - Liste der täglichen Abweichungen (float)
    """
    deviations = [0.0]  # Tag 0 ist Referenz

    for i in range(1, len(plan_list)):
        deviation = calculate_deviation_wu(
            df_original=plan_list[i - 1],
            df_new=plan_list[i],
            job_col=job_col,
            op_col=op_col,
            start_col=start_col,
            method=method
        )
        deviations.append(deviation)

    return deviations
    
def calculate_deviation_wu(df_original, df_new, job_col="Job", op_col="Operation", start_col="Start", method="sum"):
    """
    Berechnet die Abweichung der Startzeiten zwischen ursprünglichem und neuem Plan.

    Parameter:
    - method: "sum" für Gesamtabweichung, "mean" für durchschnittliche Abweichung

    Rückgabe:
    - float: gewünschte Abweichung (Summe oder Mittelwert)
    """
    merged = pd.merge(
        df_new[[job_col, op_col, start_col]],
        df_original[[job_col, op_col, start_col]],
        on=[job_col, op_col],
        suffixes=('_new', '_orig')
    )

    merged['Deviation'] = (merged[f"{start_col}_new"] - merged[f"{start_col}_orig"]).abs()

    if method == "mean":
        return merged['Deviation'].mean()
    else:
        return merged['Deviation'].sum()