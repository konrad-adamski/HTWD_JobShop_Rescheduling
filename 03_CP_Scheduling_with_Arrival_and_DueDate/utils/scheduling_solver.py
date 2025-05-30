import pandas as pd
from collections import defaultdict
from ortools.sat.python import cp_model


# Lateness ----------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# - Summe Absolute Lateness
# - Max Absolute Lateness


# Min. Summe Absolute Lateness ----------------------------------------------------------------------------------------




# Tardiness ---------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# - Summe
# - Max

def has_solution(solver, any_var):
    try:
        _ = solver.Value(any_var)
        return True
    except:
        return False


# Min. Summe Tardiness ------------------------------------------------------------------------------------------------
def solve_cp_jssp_sum_tardiness(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame, 
                                 sort_ascending: bool = False, msg: bool = False, timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:
    model = cp_model.CpModel()

    # Sortiere nach Deadline, falls gewünscht
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    # Gruppiere Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = int(row["Operation"])
            m = str(row["Machine"])
            d = int(round(row["Processing Time"]))
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    n = len(jobs)
    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # Variablen definieren
    starts, ends, intervals = {}, {}, {}
    tardiness_vars = []

    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o, (op_id, m, d) in enumerate(seq):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # Tardiness und Nebenbedingungen pro Job
    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.Add(tardiness >= job_end - deadline[job])
        tardiness_vars.append(tardiness)

        # Arrival-Bedingung
        model.Add(starts[(j, 0)] >= arrival[job])

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    # Maschinenkonflikte (NoOverlap)
    for m in machines:
        machine_intervals = [intervals[(j, o)][0] for (j, o), (interval, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # Zielfunktion: Summe der Tardiness
    model.Minimize(sum(tardiness_vars))

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.relative_gap_limit = gapRel
    solver.parameters.max_time_in_seconds = timeLimit
    status = solver.Solve(model)

    # Ergebnisse
    records = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN]:
        if has_solution(solver, next(iter(starts.values()))):
            for j, job in enumerate(jobs):
                for o, (op_id, m, d) in enumerate(all_ops[j]):
                    st = solver.Value(starts[(j, o)])
                    ed = st + d
                    records.append({
                        "Job": job,
                        "Operation": op_id,
                        "Arrival": arrival[job],
                        "Deadline": deadline[job],
                        "Machine": m,
                        "Start": st,
                        "Processing Time": d,
                        "End": ed,
                        "Tardiness": max(0, ed - deadline[job])
                    })

            df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", "Job", "Operation"]).reset_index(drop=True)
        else:
            print("No solution was found within the time limit!")
            df_schedule = pd.DataFrame()
    else:
        df_schedule = pd.DataFrame()

    print(f"\nSolver-Status: {solver.StatusName(status)}")
    if records:
        print(f"Summe Tardiness     : {solver.ObjectiveValue()}")
    print(f"Best Objective Bound: {solver.BestObjectiveBound()}")
    print(f"Laufzeit            : {solver.WallTime():.2f} Sekunden")              
 
    return df_schedule


# Min. Max Tardiness --------------------------------------------------------------------------------------------------

def solve_cp_jssp_max_tardiness(df_jssp: pd.DataFrame, df_arrivals_deadlines: pd.DataFrame, 
                                 sort_ascending: bool = False, msg: bool = False, timeLimit: int = 3600, gapRel: float = 0.0) -> pd.DataFrame:
    model = cp_model.CpModel()

    # Sortiere nach Deadline, falls gewünscht
    df_arrivals_deadlines = df_arrivals_deadlines.sort_values("Deadline", ascending=sort_ascending).reset_index(drop=True)
    arrival = df_arrivals_deadlines.set_index("Job")["Arrival"].to_dict()
    deadline = df_arrivals_deadlines.set_index("Job")["Deadline"].to_dict()
    jobs = df_arrivals_deadlines["Job"].tolist()

    # Gruppiere Operationen je Job
    ops_grouped = df_jssp.sort_values(["Job", "Operation"]).groupby("Job")
    all_ops, machines = [], set()
    for job in jobs:
        seq = []
        for _, row in ops_grouped.get_group(job).iterrows():
            op_id = int(row["Operation"])
            m = str(row["Machine"])
            d = int(round(row["Processing Time"]))
            seq.append((op_id, m, d))
            machines.add(m)
        all_ops.append(seq)

    horizon = int(df_jssp["Processing Time"].sum() + max(deadline.values()))

    # Variablen definieren
    starts, ends, intervals = {}, {}, {}
    tardiness_vars = []

    for j, job in enumerate(jobs):
        seq = all_ops[j]
        for o, (op_id, m, d) in enumerate(seq):
            suffix = f"{j}_{o}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, d, end, f"interval_{suffix}")
            starts[(j, o)] = start
            ends[(j, o)] = end
            intervals[(j, o)] = (interval, m)

    # Max-Tardiness-Variable
    max_tardiness = model.NewIntVar(0, horizon, "max_tardiness")

    for j, job in enumerate(jobs):
        last_op = len(all_ops[j]) - 1
        job_end = ends[(j, last_op)]
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{j}")
        model.Add(tardiness >= job_end - deadline[job])
        model.Add(max_tardiness >= tardiness)
        tardiness_vars.append(tardiness)

        # Arrival-Bedingung
        model.Add(starts[(j, 0)] >= arrival[job])

        # Technologische Reihenfolge
        for o in range(1, len(all_ops[j])):
            model.Add(starts[(j, o)] >= ends[(j, o - 1)])

    # Maschinenkonflikte (NoOverlap)
    for m in machines:
        machine_intervals = [intervals[(j, o)][0] for (j, o), (interval, mach) in intervals.items() if mach == m]
        model.AddNoOverlap(machine_intervals)

    # Zielfunktion: max Tardiness
    model.Minimize(max_tardiness)

    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = msg
    solver.parameters.relative_gap_limit = gapRel
    solver.parameters.max_time_in_seconds = timeLimit
    status = solver.Solve(model)

    # Ergebnisse
    records = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE, cp_model.UNKNOWN]:
        if has_solution(solver, next(iter(starts.values()))):
            for j, job in enumerate(jobs):
                for o, (op_id, m, d) in enumerate(all_ops[j]):
                    st = solver.Value(starts[(j, o)])
                    ed = st + d
                    records.append({
                        "Job": job,
                        "Operation": op_id,
                        "Arrival": arrival[job],
                        "Deadline": deadline[job],
                        "Machine": m,
                        "Start": st,
                        "Processing Time": d,
                        "End": ed,
                        "Tardiness": max(0, ed - deadline[job])
                    })

            df_schedule = pd.DataFrame.from_records(records).sort_values(["Start", "Job", "Operation"]).reset_index(drop=True)
        else:
            print("No solution was found within the time limit!")
            df_schedule = pd.DataFrame()
    else:
        df_schedule = pd.DataFrame()

    print(f"\nSolver-Status: {solver.StatusName(status)}")
    if records:
        print(f"Maximale Tardiness : {solver.ObjectiveValue()}")
    print(f"Best Objective Bound: {solver.BestObjectiveBound()}")
    print(f"Laufzeit            : {solver.WallTime():.2f} Sekunden")

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
    