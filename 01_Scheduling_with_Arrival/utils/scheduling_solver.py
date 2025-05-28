import math
import pulp
import pandas as pd
from collections import defaultdict

# Flowtime ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def solve_jssp_flowtime(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame, solver: str = 'HiGHS',
                        epsilon: float = 0.0, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert die gesamte Flow Time eines Job-Shop-Problems mit Ankunftszeiten.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - time_limit: Max. Zeit in Sekunden für den Solver.
    - epsilon: Pufferzeit in Minuten zwischen zwei Jobs auf derselben Maschine.
    - sort_ascending: Sortierung der Jobs nach Ankunft (True = früh zuerst).
    - **solver_args: Weitere Solver-Parameter wie msg=True, timeLimit= 1200, threads=5 etc.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine',
      'Start','Processing Time','Flow time','End'].
    """

    # 1. Vorverarbeitung
    df_arrivals = df_arrivals.sort_values("Arrival", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals.set_index("Job")["Arrival"].to_dict()
    jobs = df_arrivals["Job"].tolist()

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    max_arrival = max(arrival.values())
    bigM_raw = max_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

    # 2. Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops = []
    machines = set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = row["Operation"]
            m = str(row["Machine"])
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_FlowTime_Arrival", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    flow_sum = pulp.LpVariable.dicts("flowtime", jobs, lowBound=0)

    # Zielfunktion: Minimierung der Summe aller Flow Times
    prob += pulp.lpSum([flow_sum[job] for job in jobs])

    # 4. Technologische Reihenfolge + Flow-Zuordnung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

        # Letzte Operation bestimmt die Endzeit → FlowTime = End - Arrival
        d_last = seq[-1][2]
        prob += flow_sum[job] == starts[(j, len(seq) - 1)] + d_last - arrival[job]

    # 5. Maschinenkonflikte
    for m in machines:
        ops_on_m = [
            (j, o, seq[o][2])
            for j, seq in enumerate(all_ops)
            for o in range(len(seq))
            if seq[o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # 6. Solver auswählen
    solver_args.setdefault("msg", True)

    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 7. Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "Flow time": round(ed - arrival[job], 2),
                "End": round(ed, 2)
            })

    df_schedule = (
        pd.DataFrame(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Summe Flow Times        : {round(objective_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule

# gewichtet -----------------------------------------------------------------------------------------------
def solve_jssp_weighted_flowtime(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame, solver: str = 'HiGHS',
                                 epsilon: float = 0.0, sort_ascending: bool = False,
                                 **solver_args) -> pd.DataFrame:
    """
    Minimiert die gewichtete Flow Time eines Job-Shop-Problems mit Ankunftszeiten.

    Gewicht_j = sqrt(LastArrival / (1 + Arrival_j))
    Zielfunktion: sum_j Gewicht_j * (Endzeit_j - Arrival_j)

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - time_limit: Max. Zeit in Sekunden für den Solver.
    - epsilon: Pufferzeit in Minuten zwischen zwei Jobs auf derselben Maschine.
    - sort_ascending: Sortierung der Jobs nach Ankunft (True = früh zuerst).
    - **solver_args: Weitere Solver-Parameter wie msg=True, timeLimit= 1200, threads=5 etc.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine',
      'Start','Processing Time','Flow time','End'].
    """

    # 1. Vorverarbeitung
    df_arrivals = df_arrivals.sort_values("Arrival", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals.set_index("Job")["Arrival"].to_dict()
    jobs = df_arrivals["Job"].tolist()

    # Neue Gewichtung mit Wurzel
    last_arrival = max(arrival.values())
    weights = {job: math.sqrt(last_arrival / (1.0 + arrival[job])) for job in jobs}

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()
    min_arrival = min(arrival.values())
    bigM_raw = last_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw / 1000) * 1000
    print(f"BigM: {bigM}")

    # 2. Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops = []
    machines = set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = row["Operation"]
            m = str(row["Machine"])
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_WeightedFlowTime", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    flow_sum = pulp.LpVariable.dicts("flowtime", jobs, lowBound=0)

    # Zielfunktion: Gewichtete Minimierung der Flow Times mit Wurzel-Gewichten
    prob += pulp.lpSum([weights[job] * flow_sum[job] for job in jobs])

    # 4. Technologische Reihenfolge + Flow-Zuordnung
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

        # Letzte Operation bestimmt die Endzeit → FlowTime = End - Arrival
        d_last = seq[-1][2]
        prob += flow_sum[job] == starts[(j, len(seq) - 1)] + d_last - arrival[job]

    # 5. Maschinenkonflikte
    for m in machines:
        ops_on_m = [
            (j, o, seq[o][2])
            for j, seq in enumerate(all_ops)
            for o in range(len(seq))
            if seq[o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # 6. Solver auswählen
    solver_args.setdefault("msg", True)

    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)
    objective_value = pulp.value(prob.objective)

    # 7. Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "Flow time": round(ed - arrival[job], 2),
                "End": round(ed, 2)
            })

    df_schedule = (
        pd.DataFrame(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Gewichtete Flow Time     : {round(objective_value, 4)}")
    print(f"  Solver-Status            : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen         : {len(prob.variables())}")
    print(f"  Anzahl Constraints       : {len(prob.constraints)}")

    return df_schedule




# Makespan ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def solve_jssp_makespan_with_arrival(df_jssp: pd.DataFrame, df_arrivals: pd.DataFrame, solver: str = 'CBC', 
                                     epsilon: float = 0.0, sort_ascending: bool = False, **solver_args) -> pd.DataFrame:
    """
    Minimiert den Makespan eines Job-Shop-Problems mit Ankunftszeiten.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - df_arrivals: DataFrame mit ['Job','Arrival'].
    - solver: 'CBC' oder 'HiGHS' (case-insensitive).
    - time_limit: Max. Zeit in Sekunden für den Solver.
    - epsilon: Pufferzeit in Minuten zwischen zwei Jobs auf derselben Maschine.
    - sort_ascending: Sortierung der Jobs nach Ankunft (True = früh zuerst).
    - **solver_args: Weitere Solver-Parameter wie msg=True, threads=4 etc.

    Rückgabe:
    - df_schedule: DataFrame mit ['Job','Operation','Arrival','Machine',
      'Start','Processing Time','Flow time','End'].
    """

    # 1. Vorverarbeitung
    df_arrivals = df_arrivals.sort_values("Arrival", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals.set_index("Job")["Arrival"].to_dict()
    jobs = df_arrivals["Job"].tolist()

    # BigM berechnen (Worst Case)
    sum_proc_time = df_jssp["Processing Time"].sum()

    min_arrival = min(arrival.values())
    max_arrival = max(arrival.values())

    bigM_raw = max_arrival - min_arrival + sum_proc_time
    bigM = math.ceil(bigM_raw * 1.4 / 1000) * 1000
    # bigM = 1e5
    print(f"BigM: {bigM}")

    # 2. Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops = []
    machines = set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = row["Operation"]
            m = str(row["Machine"])
            d = float(row["Processing Time"])
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)

    # 3. LP-Modell
    prob = pulp.LpProblem("JSSP_Makespan_Arrival", pulp.LpMinimize)

    starts = {
        (j, o): pulp.LpVariable(f"start_{j}_{o}", lowBound=arrival[jobs[j]])
        for j in range(n)
        for o in range(len(all_ops[j]))
    }

    makespan = pulp.LpVariable("makespan", lowBound=0)
    prob += makespan

    # 4. Technologische Reihenfolge
    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o in range(1, len(seq)):
            d_prev = seq[o - 1][2]
            prob += starts[(j, o)] >= starts[(j, o - 1)] + d_prev

        # Makespan ≥ Endzeit letzte OP
        d_last = seq[-1][2]
        prob += makespan >= starts[(j, len(seq) - 1)] + d_last

    # 5. Maschinenkonflikte
    for m in machines:
        ops_on_m = [
            (j, o, seq[o][2])
            for j, seq in enumerate(all_ops)
            for o in range(len(seq))
            if seq[o][1] == m
        ]
        for i in range(len(ops_on_m)):
            j1, o1, d1 = ops_on_m[i]
            for j2, o2, d2 in ops_on_m[i + 1:]:
                if j1 == j2:
                    continue
                y = pulp.LpVariable(f"y_{j1}_{o1}_{j2}_{o2}", cat="Binary")
                prob += starts[(j1, o1)] + d1 + epsilon <= starts[(j2, o2)] + bigM * (1 - y)
                prob += starts[(j2, o2)] + d2 + epsilon <= starts[(j1, o1)] + bigM * y

    # 6. Solver auswählen
    solver_args.setdefault("msg", True)

    solver = solver.upper()
    if solver == "HIGHS":
        cmd = pulp.HiGHS_CMD(**solver_args)
    elif solver == "CBC":
        cmd = pulp.PULP_CBC_CMD(**solver_args)
    else:
        raise ValueError("Solver must be 'CBC' or 'HiGHS'")

    prob.solve(cmd)
    makespan_value = pulp.value(prob.objective)

    # 7. Ergebnisse extrahieren
    records = []
    for j, job in enumerate(jobs):
        for o, (op_id, m, d) in enumerate(all_ops[j]):
            st = starts[(j, o)].varValue
            ed = st + d
            records.append({
                "Job": job,
                "Operation": op_id,
                "Arrival": arrival[job],
                "Machine": m,
                "Start": round(st, 2),
                "Processing Time": d,
                "Flow time": round(ed - arrival[job], 2),
                "End": round(ed, 2)
            })

    df_schedule = (
        pd.DataFrame(records)
        .sort_values(["Start", "Job", "Operation"])
        .reset_index(drop=True)
    )

    # 8. Logging
    print("\nSolver-Informationen:")
    print(f"  Makespan                : {round(makespan_value, 4)}")
    print(f"  Solver-Status           : {pulp.LpStatus[prob.status]}")
    print(f"  Anzahl Variablen        : {len(prob.variables())}")
    print(f"  Anzahl Constraints      : {len(prob.constraints)}")

    return df_schedule



# FCFS --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------

def schedule_fcfs_with_arrivals(df_jssp: pd.DataFrame,
                                arrival_df: pd.DataFrame) -> pd.DataFrame:
    """
    FCFS-Scheduling mit Job-Ankunftszeiten auf Basis eines DataFrames.

    Parameter:
    - df_jssp: DataFrame mit ['Job','Operation','Machine','Processing Time'].
    - arrival_df: DataFrame mit ['Job','Arrival'].
    """
    # Arrival-Zeiten als Dict
    arrival = arrival_df.set_index('Job')['Arrival'].to_dict()

    # Status-Tracker
    next_op = {job: 0 for job in df_jssp['Job'].unique()}
    job_ready = arrival.copy()
    machine_ready = defaultdict(float)
    remaining = len(df_jssp)

    schedule = []
    while remaining > 0:
        best = None  # (job, start, dur, machine, op_idx)

        # Suche FCFS-geeignete Operation
        for job, op_idx in next_op.items():
            # Skip, wenn alle Ops geplant
            if op_idx >= (df_jssp['Job'] == job).sum():
                continue
            # Hole Row anhand Job+Operation
            row = df_jssp[(df_jssp['Job']==job)&(df_jssp['Operation']==op_idx)].iloc[0]
            m = int(row['Machine'].lstrip('M'))
            dur = row['Processing Time']
            earliest = max(job_ready[job], machine_ready[m])
            # Best-Kandidat wählen
            if (best is None or
                earliest < best[1] or
                (earliest == best[1] and arrival[job] < arrival[best[0]])):
                best = (job, earliest, dur, m, op_idx)

        job, start, dur, m, op_idx = best
        end = start + dur
        schedule.append({
            'Job': job,
            'Operation': op_idx,
            'Arrival': arrival[job],
            'Machine': f'M{m}',
            'Start': start,
            'Processing Time': dur,
            'End': end
        })
        # Update Status
        job_ready[job] = end
        machine_ready[m] = end
        next_op[job] += 1
        remaining -= 1

    df_schedule = pd.DataFrame(schedule)
    return df_schedule.sort_values(['Arrival','Start'])
    