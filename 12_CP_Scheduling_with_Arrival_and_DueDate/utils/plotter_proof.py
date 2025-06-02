import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def proof_of_concept_v1(dev_A, dev_B,
                        tardiness_A, earliness_A,
                        tardiness_B, earliness_B,
                        label_A="Strategie A",
                        label_B="Strategie B",
                        title="Proof of Concept: Rescheduling-Vergleich",
                        ylabel_left="Startzeitabweichung (Minuten)",
                        ylabel_right="Tardiness / Earliness (in %)",
                        y_right_lim=50):
    
    days = np.arange(len(dev_A))
    assert len(dev_A) == len(dev_B) == len(tardiness_A) == len(tardiness_B), "Alle Listen müssen gleich lang sein."

    # Farben
    color_bar_A = 'darkblue'
    color_bar_B = 'darkred'
    color_tard_A = 'blue'
    color_early_A = '#87CEEB'
    color_tard_B = 'red'
    color_early_B = '#FF6347'

    bar_width = 0.4

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Balken
    bars_A = ax1.bar(days - bar_width/2, dev_A, width=bar_width, label=f"{label_A} – Abweichung", color=color_bar_A)
    bars_B = ax1.bar(days + bar_width/2, dev_B, width=bar_width, label=f"{label_B} – Abweichung", color=color_bar_B)

    ax1.set_xlabel("Tag")
    ax1.set_ylabel(ylabel_left, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    for bar in bars_A:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9, color=color_bar_A)

    for bar in bars_B:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9, color=color_bar_B)

    # Zweite y-Achse (Linien)
    ax2 = ax1.twinx()
    t_A = np.array(tardiness_A) * 100
    e_A = -np.array(earliness_A) * 100
    t_B = np.array(tardiness_B) * 100
    e_B = -np.array(earliness_B) * 100

    ax2.plot(days, t_A, marker='o', color=color_tard_A, label=f"{label_A} – Tardiness")
    ax2.plot(days, e_A, marker='s', color=color_early_A, label=f"{label_A} – Earliness")
    ax2.plot(days, t_B, marker='o', linestyle='--', color=color_tard_B, label=f"{label_B} – Tardiness")
    ax2.plot(days, e_B, marker='s', linestyle='--', color=color_early_B, label=f"{label_B} – Earliness")

    ax2.set_ylabel(ylabel_right, color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(-y_right_lim, y_right_lim)

    def label_line(x, y, color):
        for xi, yi in zip(x, y):
            ax2.text(xi + 0.1, yi + 0.5, f"{yi:.1f}", ha='center', va='bottom', fontsize=8, color=color)

    label_line(days, t_A, color_tard_A)
    label_line(days, e_A, color_early_A)
    label_line(days, t_B, color_tard_B)
    label_line(days, e_B, color_early_B)

    plt.title(title)
    ax1.set_xticks(days)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Gemeinsame Legende
    lines_labels = ax1.get_legend_handles_labels()
    lines2_labels = ax2.get_legend_handles_labels()
    all_lines = lines_labels[0] + lines2_labels[0]
    all_labels = lines_labels[1] + lines2_labels[1]
    ax1.legend(all_lines, all_labels, loc='upper left', bbox_to_anchor=(1.04, 1))

    fig.tight_layout()
    plt.show()


# Combi --------------------------------------------------------------------------------------------------------------


def plot_tardiness_earliness_two_methods(tardiness_A, earliness_A,
                                         tardiness_B, earliness_B,
                                         labels=("Simple", "DevPen"),
                                         title="Vergleich der Termintreue",
                                         subtitle=None,
                                         ylabel="Ø Abweichung (Minuten)",
                                         y_lim_min=None, y_lim_max=None, as_percentage=False):
    days = np.arange(len(tardiness_A))

    # Optional in Prozent umrechnen
    factor = 100 if as_percentage else 1

    t_A = np.array(tardiness_A) * factor
    e_A = -np.array(earliness_A) * factor
    t_B = np.array(tardiness_B) * factor
    e_B = -np.array(earliness_B) * factor

    plt.figure(figsize=(10, 6))

    # Linien zeichnen
    plt.plot(days, t_A, marker='o', color='blue', label=f"{labels[0]} – Tardiness")
    plt.plot(days, e_A, marker='s', color='#87CEEB', label=f"{labels[0]} – Earliness")
    plt.plot(days, t_B, marker='o', linestyle='--', color='red', label=f"{labels[1]} – Tardiness")
    plt.plot(days, e_B, marker='s', linestyle='--', color='#FF6347', label=f"{labels[1]} – Earliness")

    # Werte beschriften
    def label_line(x, y):
        for xi, yi in zip(x, y):
            if abs(yi) > 0.01:
                plt.text(xi + 0.1, yi + 0.5 * np.sign(yi), f"{yi:.1f}", 
                         ha='center', va='bottom', fontsize=8)

    label_line(days, t_A)
    label_line(days, e_A)
    label_line(days, t_B)
    label_line(days, e_B)

    plt.xlabel("Zeit (in Tagen)")
    plt.ylabel(ylabel)
    
    full_title = title
    if subtitle:
        full_title = f"{title} {subtitle}"
    plt.title(full_title)

    plt.xticks(days)
    plt.grid(True, axis='y')

    # Y-Achsenlimits robust setzen
    if y_lim_min is not None and y_lim_max is not None:
        plt.ylim(y_lim_min, y_lim_max)
    elif y_lim_min is not None:
        plt.ylim(bottom=y_lim_min)
    elif y_lim_max is not None:
        plt.ylim(top=y_lim_max)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_two_starttime_deviation_bars(dev_A, dev_B,
                                      label_A="Strategie A",
                                      label_B="Strategie B",
                                      title="Vergleich der Startzeitabweichungen jeder Operation pro Tag",
                                      ylabel="Summe der Abweichungen",
                                      xlabel="Tag"):
    days = list(range(len(dev_A)))
    assert len(dev_A) == len(dev_B), "Beide Deviation-Listen müssen gleich lang sein."

    bar_width = 0.4

    plt.figure(figsize=(10, 6))
    bars_A = plt.bar([d - bar_width/2 for d in days], dev_A, width=bar_width, label=label_A, color='darkblue')
    bars_B = plt.bar([d + bar_width/2 for d in days], dev_B, width=bar_width, label=label_B, color='darkred')

    for bar in bars_A:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9)

    for bar in bars_B:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.1f}",
                 ha='center', va='bottom', fontsize=9)

    plt.xlabel(xlabel)
    plt.ylabel(f"{ylabel} (in Minuten)")
    plt.title(title)
    plt.xticks(days)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


# Legacy -------------------------------------------------------------------------------------------------------------


def plot_tardiness_earliness_ideal_per_day(list_tardiness_count, list_earliness_count, list_ideal=None,
                                                title="Termintreue bei Job-Ende (Final Operations)",
                                                subtitle=None,
                                                ylabel="Anteil (in %)",  y_lim = 106):
    import matplotlib.pyplot as plt
    import numpy as np

    days = np.arange(len(list_tardiness_count))

    # Umrechnung Anteil → Prozent
    values1 = np.array(list_tardiness_count) * 100
    values2 = np.array(list_earliness_count) * 100
    values3 = np.array(list_ideal) * 100 if list_ideal is not None else None

    plt.figure(figsize=(10, 6))

    plt.plot(days, values1, marker='o', label="Tardiness > 0")
    plt.plot(days, values2, marker='s', label="Earliness > 0")
    if values3 is not None:
        plt.plot(days, values3, marker='^', label="Ideal (T=0 & E=0)")

    # Werte beschriften
    def label_points(x, y):
        for xi, yi in zip(x, y):
            plt.text(xi, yi + 1, f"{yi:.1f}", ha='center', va='bottom', fontsize=9)

    label_points(days, values1)
    label_points(days, values2)
    if values3 is not None:
        label_points(days, values3)

    plt.xlabel("Zeit (in Tagen)")
    plt.ylabel(ylabel)
    if subtitle:
        title = f"{title} – {subtitle}"
    plt.title(title)
    plt.xticks(days)

    # Legende
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid(True, axis='y')
    plt.ylim(0, y_lim)
    plt.tight_layout()
    plt.show()



def plot_mean_and_max_tardiness_earliness(df_plan_last_ops_list,
                                          title="Tardiness und Earliness pro Tag",
                                          ylabel="Abweichung der Zeit (in Minuten)",
                                          subtitle=None,
                                          show_max=True):
    mean_tardiness_per_day = []
    mean_earliness_per_day = []
    max_tardiness_per_day = []
    max_earliness_per_day = []

    for df in df_plan_last_ops_list:
        mean_tardiness_per_day.append(df["Tardiness"].mean())
        mean_earliness_per_day.append(df["Earliness"].mean())
        if show_max:
            max_tardiness_per_day.append(df["Tardiness"].max())
            max_earliness_per_day.append(df["Earliness"].max())

    days = list(range(len(df_plan_last_ops_list)))

    plt.figure(figsize=(10, 6))

    # Mittelwerte
    plt.plot(days, mean_tardiness_per_day, marker='o', label='Ø Tardiness')
    plt.plot(days, mean_earliness_per_day, marker='s', label='Ø Earliness')

    # Max-Werte nur wenn gewünscht
    if show_max:
        plt.plot(days, max_tardiness_per_day, marker='^', linestyle='--', label='Max Tardiness')
        plt.plot(days, max_earliness_per_day, marker='v', linestyle='--', label='Max Earliness')

    plt.xlabel("Zeit (in Tagen)")
    plt.ylabel(ylabel)

    if subtitle:
        title = f"{title} – {subtitle}"
    plt.title(title)
    
    # Legende oben rechts im Plotbereich
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()