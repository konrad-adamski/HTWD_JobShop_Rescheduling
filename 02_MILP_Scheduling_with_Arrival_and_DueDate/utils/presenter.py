import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ausgabe der Job-Dictionary ---------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

def print_jobs(job_dict: dict):
    for job, tasks in job_dict.items():
        print(f"{job}:  {tasks}")
    print("")

# GANTT Diagramme --------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

# Farbskala: tab20 mit Überspringen von Index 6 und Layer-Anpassungen
tab20 = plt.get_cmap("tab20")

def get_color(idx):
    base_idx = idx % 16
    layer = idx // 16
    # --- Anpassung: überspringe Index 6 ---
    if base_idx >= 6:
        base_idx += 1  # 6 wird übersprungen
    rgba = tab20(base_idx / 20)  # Skaliere auf 20 Farben
    r, g, b, _ = rgba
    if layer == 1:
        r = max(0.0, r * 0.9)
        g = min(1.0, g * 1.4)
        b = max(0.0, b * 0.9)
    elif layer == 2:
        r = min(1.0, r * 1.15)
        g = max(0.0, g * 0.85)
        b = min(1.0, b * 1.15)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


# GANTT Auftragsperspektive (Job-Perspektive) 
def plot_gantt_jobs(schedule_df: pd.DataFrame, title: str = "Gantt-Diagramm", duration_column: str = "Processing Time"):
    machines = sorted(schedule_df['Machine'].unique())
    # Erzeuge Farbzurodnung pro Maschine
    color_map = {machine: get_color(i) for i, machine in enumerate(machines)}

    fig, ax = plt.subplots(figsize=(14, 8))
    jobs = sorted(schedule_df['Job'].unique())
    yticks = range(len(jobs))

    for idx, job in enumerate(jobs):
        job_ops = schedule_df[schedule_df['Job'] == job]
        for _, row in job_ops.iterrows():
            color = color_map[row['Machine']]
            ax.barh(idx,
                    row[duration_column],
                    left=row['Start'],
                    height=0.5,
                    color=color,
                    edgecolor='black')

    # Legende erstellen
    legend_handles = [mpatches.Patch(color=color_map[m], label=str(m)) for m in machines]
    ax.legend(handles=legend_handles,
              title="Maschinen",
              bbox_to_anchor=(1.05, 1),
              loc='upper left')

    ax.set_yticks(yticks)
    ax.set_yticklabels(jobs)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Jobs")
    ax.set_title(title)
    ax.grid(True)

    # Achsenlimits
    max_time = (schedule_df['Start'] + schedule_df[duration_column]).max()
    ax.set_xlim(0, max_time * 1.05)
    plt.tight_layout()
    plt.show()



# GANTT Maschinenperspektive
def plot_gantt_machines(schedule_df: pd.DataFrame, title: str = "Gantt-Diagramm (Maschinenansicht)", duration_column: str = "Processing Time"):
    machines = sorted(schedule_df['Machine'].unique())
    jobs = sorted(schedule_df['Job'].unique())
    yticks = range(len(machines))

    # Erzeuge Farbzuordnung pro Job
    color_map = {job: get_color(i) for i, job in enumerate(jobs)}

    # Dynamische Höhe: 0.8 Inch pro Maschine
    fig_height = len(machines) * 0.8
    fig, ax = plt.subplots(figsize=(16, fig_height))

    for idx, machine in enumerate(machines):
        ops = schedule_df[schedule_df['Machine'] == machine]
        for _, row in ops.iterrows():
            ax.barh(idx,
                    row[duration_column],
                    left=row['Start'],
                    height=0.5,
                    color=color_map[row['Job']],
                    edgecolor='black')

    # Legende (Jobs)
    legend_handles = [mpatches.Patch(color=color_map[j], label=str(j)) for j in jobs]
    legend_columns = (len(jobs) // 35) + 1
    ax.legend(handles=legend_handles,
              title="Jobs",
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              ncol=legend_columns)

    ax.set_yticks(yticks)
    ax.set_yticklabels(machines)
    ax.set_xlabel("Minuten")
    ax.set_ylabel("Maschinen")
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Achsenlimits
    max_time = (schedule_df['Start'] + schedule_df[duration_column]).max()

    x_start = int((schedule_df['Start'].min() // 1440) * 1440)
    ax.set_xlim(x_start, max_time + 120)

    # Vertikale Linien alle 1440 Einheiten (z.B. 1 Tag bei Minuten)
    for x in range(x_start, int(max_time) + 1440, 1440):
        ax.axvline(x=x, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.show()


# Dataframe gruppiert ----------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------

def count_column_grouped(df: pd.DataFrame, column: str = 'Tardiness',
                         bins: list = [-np.inf, 0, 30, 60, 120, 240, 480, 720, 1440, 2880, np.inf], right_closed: bool = False) -> pd.Series:
    """
    Zählt Werte in definierten Intervallen, trennt Null-Werte explizit und
    erfasst auch negative Werte (<0) separat.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        column (str): Zu gruppierende Spalte
        bins (list): Intervallgrenzen, inkl. -inf und inf
        right_closed (bool): Ob die rechte Grenze inklusiv ist (Standard: linksoffen)

    Returns:
        pd.Series: Gezählte Werte mit Intervall-Indizes,
                   z. B. '<0', '0', '0-100', '100-200', '200-500', '>500'
    """
    # 1. Input-Validierung
    if column not in df.columns:
        raise ValueError(f"Spalte '{column}' existiert nicht. Verfügbare Spalten: {list(df.columns)}")

    # 2. Null-Werte separat zählen
    zero_count = (df[column] == 0).sum()

    # 3. Nur Nicht-Null-Werte für bins
    non_zero = df.loc[df[column] != 0, column]

    # 4. Dynamische Label-Erzeugung aus den Bin-Grenzen
    labels = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isneginf(lo):
            labels.append('<' + str(int(hi)) if not np.isposinf(hi) else '<inf')
        elif np.isposinf(hi):
            labels.append('>' + str(int(lo)))
        else:
            labels.append(f"{int(lo)}-{int(hi)}")
    # Beispiel: ['<0', '0-100', '100-200', '200-500', '>500']

    # 5. Gruppierung für alle Nicht-Null-Werte
    grouped = pd.cut(non_zero, bins=bins, labels=labels, right=right_closed, include_lowest=True)

    # 6. Zählung pro Intervall
    counts = grouped.value_counts().reindex(labels, fill_value=0)

    # 7. Zero-Wert-Count einfügen
    #    Hier setzen wir das Label '0' zwischen '<0' und '0-100'
    result = pd.concat([
        pd.Series({labels[0]: counts.iloc[0]}),  # '<0'
        pd.Series({'0': zero_count}),
        counts.iloc[1:].rename_axis(None)
    ]).astype(int)

    return result


