"""The side-operator wiring checker (O8) against task graphs built by a model of the runtime's
register_mugraph rules: tasks in registration order, one launch event per chained pair whose
range covers the consumer's tasks, all launch events turned into empty counter events whose
range assigns the dependent events, the last operator's tasks (and the side tasks) triggering
the end-of-graph event. The checker reads what a VM build writes (task_graph_rank0.json)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fleet import task_graph_check as C  # noqa: E402

BEGIN, END, EMPTY = 903, 910, 900
INVALID = C.EVENT_INVALID_ID


def model_graph(ops, side_types=(193,)):
    """ops: [(task_type, n_tasks, side)] in registration order. Returns the task graph dict the
    runtime would write, per its rules (with the side-operator branch of the patch)."""
    tasks = [{"task_type": 100, "trigger_event": 0, "dependent_event": INVALID}]     # begin task, event 0
    events = [{"event_type": BEGIN, "num_triggers": 1, "first_task_id": 2, "last_task_id": 0}]
    pre = None            # the host's task ids
    pre_first = False
    side_ids = []
    for ttype, n, side in ops:
        ids = list(range(len(tasks), len(tasks) + n))
        for _ in ids:
            tasks.append({"task_type": ttype, "trigger_event": INVALID, "dependent_event": INVALID})
        if side:
            side_ids += ids
            if not pre_first:
                events[-1]["last_task_id"] = len(tasks)          # the host's event range grows
            continue
        if pre is None:
            pre_first = True
        else:
            pre_first = False
            e = len(events)
            for t in pre:
                tasks[t]["trigger_event"] = e
            events.append({"event_type": EMPTY, "num_triggers": len(pre), "first_task_id": ids[0],
                           "last_task_id": ids[-1] + 1})
        pre = ids
    e = len(events)
    for t in pre + side_ids:
        tasks[t]["trigger_event"] = e
    events.append({"event_type": END, "num_triggers": len(pre) + len(side_ids), "first_task_id": 0, "last_task_id": 0})
    events[0]["last_task_id"] = len(tasks)
    for i, ev in enumerate(events):
        if ev["event_type"] == EMPTY:
            for t in range(ev["first_task_id"], ev["last_task_id"]):
                tasks[t]["dependent_event"] = i
    return {"all_tasks": tasks, "all_events": events, "first_tasks": []}


def test_side_tasks_after_a_chained_host_share_its_event_and_trigger_the_end():
    tg = model_graph([(101, 1, False), (102, 4, False), (193, 3, True), (103, 2, False), (194, 5, True)])
    problems, s = C.check_side_tasks(tg)
    assert problems == [], problems
    assert s["side_tasks"] == 8 and s["end_num_triggers"] == 2 + 8
    assert s["task_types"] == {"100": 1, "101": 1, "102": 4, "193": 3, "103": 2, "194": 5}   # the counts per type, in type order
    # the side tasks after op 102 carry op 102's dependent event, those after 103 carry 103's
    tasks = tg["all_tasks"]
    assert tasks[6]["dependent_event"] == tasks[2]["dependent_event"] != INVALID
    assert tasks[10]["dependent_event"] == tasks[9]["dependent_event"]


def test_side_tasks_after_the_first_operator_have_no_dependent_event():
    tg = model_graph([(101, 1, False), (193, 2, True), (102, 3, False)])
    problems, s = C.check_side_tasks(tg)
    assert problems == [] and s["end_num_triggers"] == 3 + 2
    assert tg["all_tasks"][2]["dependent_event"] == INVALID == tg["all_tasks"][1]["dependent_event"]


def test_the_checker_catches_a_side_task_left_on_the_chain():
    tg = model_graph([(101, 1, False), (102, 4, False), (193, 3, True), (103, 2, False)])
    # a side task wired as if it were chained: its own event and no end trigger
    t = tg["all_tasks"][7]
    t["trigger_event"] = 1
    problems, _ = C.check_side_tasks(tg)
    assert any("trigger is not the end-of-graph" in p for p in problems)
    assert any("num_triggers" in p for p in problems)
    tg = model_graph([(101, 1, False), (102, 4, False), (193, 3, True), (103, 2, False)])
    tg["all_tasks"][7]["dependent_event"] = 99
    problems, _ = C.check_side_tasks(tg)
    assert any("differs from its host" in p for p in problems)


def test_cli_reports_pass(tmp_path, capsys):
    import json
    f = tmp_path / "task_graph_rank0.json"
    f.write_text(json.dumps(model_graph([(101, 1, False), (102, 2, False), (193, 2, True)])))
    assert C.main([str(f)]) == 0
    assert "PASS" in capsys.readouterr().out
