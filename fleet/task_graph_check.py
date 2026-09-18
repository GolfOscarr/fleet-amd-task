#!/usr/bin/env python3
"""Check the side operators' wiring in a compiled task graph (O8,
docs/gpu-experiments/03-acceleration/03-local-preparation.md).

The runtime writes task_graph_rank0.json next to the build (persistent_kernel.py
compile(); harness/run_fleet.py keeps it under the run's build directory): a list
of tasks (task_type, trigger_event, dependent_event) and of events (event_type,
num_triggers, first_task_id, last_task_id). The side-operator branch of the
runtime patch must have given every side task (types 193 and 194)

  - the dependent event of the task registered just before it (its host's), or
    no dependent event when the host is the graph's first operator;
  - the end-of-graph event as its trigger; and
  - the end-of-graph event's num_triggers equal to the number of tasks that
    trigger it (the last operator's plus the side tasks').

    python fleet/task_graph_check.py <run>/build/task_graph_rank0.json
"""
import json
import sys

SIDE_TYPES = (193, 194)          # TASK_PREFETCH_MI300, TASK_PREFETCH_MOE_MI300
EVENT_END_OF_TASK_GRAPH = 910
EVENT_INVALID_ID = 0x7FFFFFFFFFFFFFFE


def event_index(event_id):
    """The position of an event id in all_events (the low 32 bits; the gpu id sits above)."""
    return event_id & 0xFFFFFFFF


def check_side_tasks(tg, side_types=SIDE_TYPES):
    """Returns the list of problems (empty when the wiring is right) and a summary dict."""
    tasks, events = tg["all_tasks"], tg["all_events"]
    problems = []
    end_events = [i for i, e in enumerate(events) if e["event_type"] == EVENT_END_OF_TASK_GRAPH]
    if len(end_events) != 1:
        return [f"expected one end-of-graph event, found {len(end_events)}"], {}
    end = end_events[0]
    side = [i for i, t in enumerate(tasks) if t["task_type"] in side_types]
    for i in side:
        t = tasks[i]
        if t["trigger_event"] == EVENT_INVALID_ID or event_index(t["trigger_event"]) != end:
            problems.append(f"task {i} (type {t['task_type']}): trigger is not the end-of-graph event")
        # the host: the nearest earlier task that is not a side task
        j = i - 1
        while j >= 0 and tasks[j]["task_type"] in side_types:
            j -= 1
        if j < 0:
            problems.append(f"task {i}: no host task before it")
            continue
        if tasks[j]["dependent_event"] != t["dependent_event"]:
            problems.append(f"task {i}: dependent event {t['dependent_event']} differs from its host task {j}'s "
                            f"{tasks[j]['dependent_event']}")
    triggers_end = sum(1 for t in tasks if t["trigger_event"] != EVENT_INVALID_ID
                       and event_index(t["trigger_event"]) == end)
    if events[end]["num_triggers"] != triggers_end:
        problems.append(f"end-of-graph num_triggers {events[end]['num_triggers']} but {triggers_end} tasks trigger it")
    # every side task sits inside the launch range of the event it depends on (the host's extended range)
    for i in side:
        dep = tasks[i]["dependent_event"]
        if dep == EVENT_INVALID_ID:
            continue
        if event_index(dep) >= len(events):
            problems.append(f"task {i}: dependent event {dep} is not an event of the graph")
            continue
        e = events[event_index(dep)]
        if not (e["first_task_id"] <= i < e["last_task_id"]):
            problems.append(f"task {i}: outside the range [{e['first_task_id']}, {e['last_task_id']}) of its event")
    # round 4 (docs/gpu-experiments/04-kernels/07-session-plan.md, the tgcheck rows): the task count
    # per type, read against the plan's (195 the GEMV linear, 196 the gang w13 GEMV, 204 the
    # four-task router, 205 the regular merge, 207 the merge with o_proj folded in)
    types = {}
    for t in tasks:
        types[t["task_type"]] = types.get(t["task_type"], 0) + 1
    summary = {"tasks": len(tasks), "events": len(events), "side_tasks": len(side),
               "end_event": end, "end_num_triggers": events[end]["num_triggers"],
               "task_types": {str(k): v for k, v in sorted(types.items())}}
    return problems, summary


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    tg = json.loads(open(argv[0]).read())
    problems, summary = check_side_tasks(tg)
    print(json.dumps(summary))
    for p in problems:
        print("PROBLEM", p)
    print("PASS" if not problems else "FAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
