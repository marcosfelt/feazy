"""
Schedule a list of tasks
"""
from multiprocessing.connection import wait
from feazy.plan.utils import Graph, GraphSearch
from .task import Task, TaskGraph
from gcsa.google_calendar import GoogleCalendar, Event
from gcsa.event import Transparency
from beautiful_date import *
from ortools.sat.python import cp_model
from datetime import datetime, time, timedelta, date
from typing import List, Union, Tuple, Optional, Dict
from queue import LifoQueue, SimpleQueue
import pytz
import math
import numpy as np
from copy import deepcopy
import logging
import collections
import re

from feazy.plan import task

DEFAULT_TIMEZONE = "Europe/London"

fmt_date = lambda d: d.astimezone(pytz.timezone(DEFAULT_TIMEZONE)).strftime(
    "%m/%d/%Y, %H:%M:%S"
)


def filter_availabilities(
    availabilities: List[Tuple[datetime, datetime]],
    work_times: Dict[int, Tuple[time, time]],
    work_timezone: str = DEFAULT_TIMEZONE,
) -> List[Tuple[datetime, datetime]]:
    """Filter availabilities to work times

    Arguments
    --------
    availabilities : List[Tuple[time, time]]
        A list of start and end times of each available block.
    work_times : Dict[int, Tuple[time, time]]
        A dictionary with keys as days of the week by integer starting with Monday as 0.
        Values should be tuples with the start and end time of work on that day.
    work_timezone : string
        Timezone of the work times

    Returns
    -------
    new_availabilities : List[Tuple[time, time]]
        List of availabilities filtered to work times
    """

    new_availabilities = []
    # Convert everything to the work timezone
    tz = pytz.timezone(work_timezone)
    availabilities = [
        (a[0].astimezone(tz), a[1].astimezone(tz)) for a in availabilities
    ]

    for availability in availabilities:
        n_days = math.ceil(
            (availability[1] - availability[0]).total_seconds() / (3600 * 24)
        )
        current_availability = availability
        while n_days > 0:
            work_day = work_times.get(current_availability[0].weekday())
            skip_current_day = False
            if work_day is not None:
                # Move availabilities starting before working hours to beginning of work time
                if current_availability[0].time() < work_day[0]:
                    current_availability = (
                        datetime.combine(
                            current_availability[0].date(),
                            work_day[0],
                            tzinfo=current_availability[1].tzinfo,
                        ),
                        current_availability[1],
                    )
                # If availability starts after work time, skip
                elif current_availability[0].time() > work_day[1]:
                    skip_current_day = True

                # If on final day and current availability ends after specified end, move back to specified
                final_day = current_availability[1].date() == availability[1].date()
                if (
                    final_day
                    and current_availability[1].time() > availability[1].time()
                ):
                    current_availability = (
                        current_availability[0],
                        datetime.combine(
                            availability[1].date(),
                            availability[1].time(),
                            tzinfo=availability[1].tzinfo,
                        ),
                    )

                same_day = (
                    current_availability[0].date() == current_availability[1].date()
                )
                # If availability ends before work time begins on the same day, move on
                if same_day and (current_availability[1].time() < work_day[0]):
                    skip_current_day = True
                # If availabiltiy ends after work time on the same day, move it to the end of work time
                elif same_day and current_availability[1].time() > work_day[1]:
                    current_availability = (
                        current_availability[0],
                        datetime.combine(
                            current_availability[1].date(),
                            work_day[1],
                            tzinfo=current_availability[1].tzinfo,
                        ),
                    )
                if not same_day:
                    current_availability = (
                        current_availability[0],
                        datetime.combine(
                            current_availability[0].date(),
                            work_day[1],
                            tzinfo=current_availability[0].tzinfo,
                        ),
                    )

                if not skip_current_day:
                    new_availabilities.append(current_availability)

            # Move to next day
            next_day = current_availability[0].date() + timedelta(days=1)
            new_times = work_times.get(
                next_day.weekday(),
                [
                    current_availability[0].time(),
                    current_availability[1].time(),
                ],
            )
            current_availability = (
                tz.localize(
                    datetime.combine(
                        next_day,
                        new_times[0],
                    )
                ),
                tz.localize(
                    datetime.combine(
                        next_day,
                        new_times[1],
                    )
                ),
            )

            # Decrement number days
            n_days -= 1
    return new_availabilities


def find_start_tasks(tasks: TaskGraph) -> List[Task]:
    """Find start tasks using the adjancency matrix

    I'm sure there's a faster way to do this but this made sense to me
    """
    n_tasks = len(tasks._nodes)
    adjacency_matrix = np.zeros([n_tasks, n_tasks])
    task_to_index = {task.task_id: i for i, task in enumerate(tasks.all_tasks)}
    index_to_task = {i: task.task_id for i, task in enumerate(tasks.all_tasks)}
    for task in tasks.all_tasks:
        for adj in task.adjacents:
            adjacency_matrix[
                task_to_index[task.task_id], task_to_index[adj.task_id]
            ] = 1
    start_tasks = []
    for i in range(n_tasks):
        if adjacency_matrix[:, i].sum() == 0:
            start_tasks.append(tasks[index_to_task[i]])
    return start_tasks


def breakdown_tasks(tasks: TaskGraph, block_duration: timedelta) -> TaskGraph:
    """Breakdown tasks into smaller blocks"""
    new_tasks = deepcopy(
        tasks
    )  # New tasks graph for inserting breaks and broken down blocks

    # Breakdown tasks
    for current_task in tasks.all_tasks:
        # Break up task if necessary
        if current_task.duration > block_duration:
            n_blocks = int(math.ceil(current_task.duration / block_duration))
            new_tasks[current_task.task_id].duration = block_duration
            new_tasks[current_task.task_id].description += f" (Block 1/{n_blocks})"
            wait_time = current_task.wait_time
            new_tasks[current_task.task_id].wait_time = timedelta()
            last_block = new_tasks[current_task.task_id]
            for b in range(n_blocks - 1):
                new_block = Task(
                    task_id=current_task.task_id + f"_{b+2}",
                    duration=block_duration,
                    description=current_task.description + f" (Block {b+2}/{n_blocks})",
                    earliest_start=current_task.earliest_start,
                    deadline=current_task.deadline,
                )
                new_tasks.add_task(new_block)
                new_tasks.add_dependency(last_block.task_id, new_block.task_id)
                last_block = new_block
            # Connect last block to successors
            for adj in current_task.adjacents:
                new_tasks.remove_dependency(current_task.task_id, adj.task_id)
                new_tasks.add_dependency(last_block.task_id, adj.task_id)
            # Set wait time to be after the last block
            last_block.wait_time = wait_time

    for task in new_tasks.all_tasks:
        if task.duration > block_duration:
            raise ValueError(
                f"""Duration of task "{task.description}" ({timedelta.total_seconds()/3600} hr) is greater than block duration ({block_duration.total_seconds()/3600}hr) """
            )

    assert not new_tasks.is_cyclic()

    start_tasks = find_start_tasks(new_tasks)
    for task in start_tasks:
        visited = [task]
        res = recursive_check(task, visited)
        if res:
            raise ValueError(f"Recursive check failed with {res}")
    return new_tasks


def consolidate_tasks(original_tasks: TaskGraph, block_tasks: TaskGraph) -> TaskGraph:
    """Consolidate tasks after breadking them down"""
    new_tasks = deepcopy(original_tasks)
    for task in new_tasks.all_tasks:
        task.scheduled_early_start = None
        task.scheduled_early_finish = None
        task.scheduled_late_start = None
        task.scheduled_deadline = None

    # Consolidate tasks
    for current_task in block_tasks.all_tasks:
        if not all(
            [
                current_task.scheduled_early_start,
                current_task.scheduled_late_start,
                current_task.scheduled_early_finish,
                current_task.scheduled_deadline,
            ]
        ):
            continue
        task_id = current_task.task_id
        split = task_id.split("_")
        if len(split) > 1:
            id = split[0]
        else:
            id = task_id

        # Scheduled early start
        if new_tasks[id].scheduled_early_start is not None:
            if current_task.scheduled_early_start < new_tasks[id].scheduled_early_start:
                new_tasks[id].scheduled_early_start = raise_none(
                    current_task.scheduled_early_start
                )
        else:
            new_tasks[id].scheduled_early_start = raise_none(
                current_task.scheduled_early_start
            )

        # Scheduled late start
        if new_tasks[id].scheduled_late_start is not None:
            if current_task.scheduled_late_start < new_tasks[id].scheduled_late_start:
                new_tasks[id].scheduled_late_start = raise_none(
                    current_task.scheduled_late_start
                )
        else:
            new_tasks[id].scheduled_late_start = raise_none(
                current_task.scheduled_late_start
            )
        raise_none(new_tasks[id].scheduled_late_start)

        # Scheduled finish
        if new_tasks[id].scheduled_early_finish is not None:
            if (
                current_task.scheduled_early_finish
                > new_tasks[id].scheduled_early_finish
            ):
                new_tasks[id].scheduled_early_finish = raise_none(
                    current_task.scheduled_early_finish
                )
        else:
            new_tasks[id].scheduled_early_finish = raise_none(
                current_task.scheduled_early_finish
            )
        raise_none(new_tasks[id].scheduled_early_finish)

        # Scheduled deadline
        if new_tasks[id].scheduled_deadline is not None:
            if current_task.scheduled_deadline > new_tasks[id].scheduled_deadline:
                new_tasks[id].scheduled_deadline = raise_none(
                    current_task.scheduled_deadline
                )
        else:
            new_tasks[id].scheduled_deadline = raise_none(
                current_task.scheduled_deadline
            )
        raise_none(new_tasks[id].scheduled_deadline)
    return new_tasks


def raise_none(val):
    if not val:
        print(f"Missing value: {val}")
    else:
        return val


def optimize_timing(
    tasks: TaskGraph,
    availabilities: List[Tuple[datetime, datetime]],
    block_duration: timedelta,
    start_time: Optional[datetime] = None,
    deadline: Optional[datetime] = None,
    max_solutions: Optional[int] = 10,
    timeout: Optional[float] = 60.0,
    separate_objectives=False,
) -> TaskGraph:
    """Optimize timing using CpSAT solver"""
    logger = logging.getLogger(__name__)

    # Use beginning and end of availability as start and end time by default
    if start_time is None:
        start_time = availabilities[0][0]
    if deadline is None:
        deadline = availabilities[-1][0]

    # Split tasks into blocks
    task_blocks = breakdown_tasks(tasks, block_duration=block_duration)
    logger.debug(
        f"Number of tasks to schedule (after breaking down): {len(task_blocks.all_tasks)}."
    )

    # Formulate and solve optimization problem
    # First try fixing tasks that are already in Gtasks/Reclaim
    # If that fails, unfix and try retry solving
    for fix_status in [False]:
        (
            model,
            early_vars,
            late_vars,
            presences,
            late_finish_obj_var,
            tasks_completed_obj,
            tasks_completed,
            sum_task_slacks,
        ) = create_optimization_model_time_based(
            task_blocks,
            availabilities,
            start_time,
            deadline,
            set_objective=not separate_objectives,
            fix_gtasks=fix_status,
        )
        solver, status = solve_cpsat(
            model,
            early_vars=early_vars,
            max_solutions=max_solutions,
            finish=late_finish_obj_var,
            task_completed_obj=tasks_completed_obj,
            tasks_completed=tasks_completed,
            sum_task_slacks=sum_task_slacks,
            set_objective=separate_objectives,
            timeout=timeout,
        )

        # Update tasks with scheduled start times and deadlines
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            new_tasks = post_process_solution(
                solver=solver,
                tasks=task_blocks,
                start_time=start_time,
                early_vars=early_vars,
                late_vars=late_vars,
                presences=presences,
                fix_gtasks=fix_status,
            )

            # Consolidate tasks
            consolidated_tasks = consolidate_tasks(
                original_tasks=tasks, block_tasks=new_tasks
            )
            return consolidated_tasks
    raise ValueError("No solution found")


def convert_datetime_to_model_hours(start_time: datetime, time: datetime) -> int:
    return int((time - start_time).total_seconds() / 3600)


def convert_model_hours_to_datetime(start_time: datetime, hours: int) -> datetime:
    return start_time + timedelta(hours=hours)


task_type = collections.namedtuple("task_type", "start end interval")


def create_optimization_model_time_based(
    tasks: TaskGraph,
    availabilities: List[Tuple[datetime, datetime]],
    start_time: datetime,
    deadline: datetime,
    set_objective=False,
    optimize_early_finish=False,
    fix_gtasks=False,
) -> Tuple[cp_model.CpModel, Dict, Dict]:
    """Create OR-tools optimization model. Time is in hours"""
    logger = logging.getLogger(__name__)
    # Create model
    model = cp_model.CpModel()

    # Create variables for early and late starts + intervals
    early_vars = {}
    late_vars = {}
    presences: Dict[str, cp_model.IntVar] = {}
    project_deadline_hours = convert_datetime_to_model_hours(start_time, deadline)
    first_availability = convert_datetime_to_model_hours(
        start_time, availabilities[0][0]
    )
    logger.debug(f"Project deadline hours: {project_deadline_hours}")
    for task in tasks.all_tasks:
        presence = model.NewBoolVar(f"presence_{task.task_id}")
        presences[task.task_id] = presence
        for name, var_group in {
            "early": early_vars,
            "late": late_vars,
        }.items():
            if task.gtasks_id and fix_gtasks and task.scheduled_early_start:
                start = convert_datetime_to_model_hours(
                    start_time,
                    task.scheduled_early_start
                    if name == "early"
                    else task.scheduled_late_start,
                )
                dur = int(task.duration.total_seconds() / 3600)
                interval_var = model.NewOptionalFixedSizeIntervalVar(
                    start, dur, presence, f"{name}_interval_{task.task_id}"
                )
                var_group[task.task_id] = task_type(
                    start=start,
                    end=start + dur,
                    interval=interval_var,
                )
            else:
                lb = (
                    0
                    if task.earliest_start is None
                    else convert_datetime_to_model_hours(
                        start_time, task.earliest_start
                    )
                )
                if lb < 0:
                    raise ValueError(
                        f"""Task "{task.description}" starts before start time."""
                    )
                ub = (
                    project_deadline_hours
                    if task.deadline is None
                    else convert_datetime_to_model_hours(start_time, task.deadline)
                )
                if ub > project_deadline_hours:
                    raise ValueError(
                        f"""Task "{task.description}" deadline {fmt_date(task.deadline)} is greater than project deadline {deadline}."""
                    )
                start_var = model.NewIntVar(lb, ub, f"{name}_start_{task.task_id}")
                dur = int(task.duration.total_seconds() / 3600)

                interval_var = model.NewOptionalFixedSizeIntervalVar(
                    start_var, dur, presence, f"{name}_interval_{task.task_id}"
                )

                var_group[task.task_id] = task_type(
                    start=start_var,
                    end=start_var + dur,
                    interval=interval_var,
                )

    # Define non-available intervals
    availabilities.sort(key=lambda a: a[0], reverse=False)
    non_available_intervals = []
    if availabilities[0][0] != start_time:
        non_available_intervals.append(
            model.NewFixedSizeIntervalVar(0, first_availability, "busy_interval_0")
        )
    for i in range(1, len(availabilities)):
        start = convert_datetime_to_model_hours(start_time, availabilities[i - 1][1])
        end = convert_datetime_to_model_hours(start_time, availabilities[i][0])
        size = end - start
        # Check for no overlapping availabilities
        if i >= 2:
            check = availabilities[i - 1][0] > availabilities[i - 2][1]
            if not check:
                raise ValueError(
                    f"{fmt_date(availabilities[i - 1][0])} does not come after {fmt_date(availabilities[i - 2][1])}"
                )
        if size > 0:
            non_available_intervals.append(
                model.NewFixedSizeIntervalVar(start, size, f"busy_interval_{i}")
            )

    # No overlap constraint with tasks and availability constraints
    for var_group in [early_vars, late_vars]:
        model.AddNoOverlap(
            [group.interval for group in var_group.values()] + non_available_intervals
        )

    # Only allow scheduling all tasks in a block
    task_blocks = {}
    for task in tasks.all_tasks:
        task_id = task.task_id
        split = task_id.split("_")
        if len(split) > 1:
            id = split[0]
        else:
            continue
        block = task_blocks.get(id)
        if block is None:
            task_blocks[id] = [task_id]
        else:
            task_blocks[id].append(task_id)
    for id, task_block in task_blocks.items():
        model.Add(
            presences[id] * len(task_block)
            == sum(presences[block_id] for block_id in task_block)
        )  # All or nothing

    # Precedence constraint with wait times
    for task in tasks.all_tasks:
        for var_group in [early_vars, late_vars]:
            for succ in task.successors:
                wait_time = int(task.wait_time.total_seconds() / 3600)
                model.Add(
                    var_group[task.task_id].end + wait_time
                    <= var_group[succ.task_id].start
                ).OnlyEnforceIf(presences[task.task_id])

    # Slack constraint (early start must before late start)
    slacks = {}
    for task in tasks.all_tasks:
        slack = model.NewIntVar(0, project_deadline_hours, f"slack_{task.task_id}")
        model.Add(
            slack == late_vars[task.task_id].start - early_vars[task.task_id].start
        ).OnlyEnforceIf(presences[task.task_id])
        model.Add(slack == 0).OnlyEnforceIf(presences[task.task_id].Not())
        slacks[task.task_id] = slack

    # Objectives: Maximize number of tasks completed and sum of slacks while minimizing finish time
    # Completed tasks
    tasks_completed_obj = calculate_depth_weighted_completion_objective(
        tasks, presences
    )
    task_completed = sum(presences[task.task_id] for task in tasks.all_tasks)
    # Finish time
    if optimize_early_finish:
        finish_obj_var = model.NewIntVar(0, project_deadline_hours, "early_finish")
        for task in tasks.all_tasks:
            model.Add(early_vars[task.task_id].end <= finish_obj_var).OnlyEnforceIf(
                presences[task.task_id]
            )
    else:
        finish_obj_var = model.NewIntVar(0, project_deadline_hours, "late_finish")
        for task in tasks.all_tasks:
            model.Add(late_vars[task.task_id].end <= finish_obj_var).OnlyEnforceIf(
                presences[task.task_id]
            )
    # Slack -> Only take slacks for base tasks
    sum_task_slacks = sum(
        [slack for task_id, slack in slacks.items() if len(task_id.split("_")) == 1]
    )
    if set_objective:
        model.Minimize(1e5 * finish_obj_var - tasks_completed_obj - sum_task_slacks)

    return (
        model,
        early_vars,
        late_vars,
        presences,
        finish_obj_var,
        tasks_completed_obj,
        task_completed,
        sum_task_slacks,
    )


def calculate_max_depth(tasks: TaskGraph, start_tasks: List[Task]):
    max_depth = 0
    for start_task in start_tasks:
        depth = 0
        for task in tasks.graph_search(start_task.task_id, GraphSearch.DFS):
            depth += 1
            if len(task.successors) == 0:
                if depth > max_depth:
                    max_depth = depth
                break
    return max_depth


def calculate_depth_weighted_completion_objective(
    tasks: TaskGraph, presences: Dict[str, cp_model.IntVar]
):
    "Calculate number of tasks completed inversely weighted by task depth"
    start_tasks = find_start_tasks(tasks)
    max_depth = calculate_max_depth(tasks, start_tasks)
    tasks_completed = []
    visited = {}
    for start_task in start_tasks:
        visit_list = SimpleQueue()

        # Add root to queue to start
        root = tasks[start_task.task_id]
        if root:
            visit_list.put(root)
        else:
            ValueError(f"{start_task.task_id} is not in the graph")

        next_level = 0
        current_level = 1
        depth = 1
        while not visit_list.empty():
            task = visit_list.get(block=True)
            if task.val not in visited:
                # Add depth weighted task completion
                coeff = int(max_depth / depth)
                tasks_completed.append(coeff * presences[task.task_id])
                # Update variables
                visited[task.val] = task
                for adjacent in task.successors:
                    visit_list.put(adjacent)
                next_level += len(task.successors)
                current_level -= 1
                if current_level <= 0:
                    current_level = next_level
                    depth += 1
    # ensure all tasks were visited
    for task in tasks.all_tasks:
        if not task.task_id in visited:
            raise ValueError(
                f"{task.task_id} not incluced depth weighted completion calculation"
            )
    return sum(tasks_completed)


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(
        self,
        max_solutions: int,
        finish: cp_model.IntVar,
        sum_task_slacks: cp_model.IntVar,
        tasks_completed_obj: cp_model.IntVar,
        tasks_completed: cp_model.IntVar,
        total_tasks: int,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._solution_count = 0
        self._solution_limit = max_solutions
        self._finish = finish
        self._sum_task_slacks = sum_task_slacks
        self._tasks_completed_obj = tasks_completed_obj
        self._tasks_completed = tasks_completed
        self._total_tasks = total_tasks

    def OnSolutionCallback(self):
        return self.on_solution_callback()

    def on_solution_callback(self):
        self._solution_count += 1

        late_finish = self.Value(self._finish)
        sum_task_slacks = self.Value(self._sum_task_slacks)
        tasks_completed = self.Value(self._tasks_completed)
        task_completed_obj = self.Value(self._tasks_completed_obj)

        print(
            f"# Tasks scheduled: {tasks_completed}/{self._total_tasks} (objective: {task_completed_obj}) | Finish objective: {late_finish} | Total task slacks: {sum_task_slacks}"
        )

        if self._solution_count >= self._solution_limit:
            print(f"Stop search after {self._solution_count} solutions.")
            self.StopSearch()

    @property
    def solution_count(self):
        return self._solution_count

    def reset(self):
        self._solution_count = 0


def solve_cpsat(
    model: cp_model.CpModel,
    finish: cp_model.IntVar,
    tasks_completed: cp_model.IntVar,
    task_completed_obj: cp_model.IntVar,
    sum_task_slacks: cp_model.IntVar,
    early_vars: Dict[str, task_type],
    max_solutions=10,
    num_workers=6,
    log_search_progress=True,
    set_objective=False,
    timeout=60,
) -> None:
    """Solve optimization problem using CpSAT"""
    logger = logging.getLogger(__name__)
    logger.debug("Solving optimization problem using CPSat")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = log_search_progress
    solver.parameters.num_search_workers = num_workers
    solver.parameters.max_time_in_seconds = timeout

    cb = SolutionCallback(
        max_solutions=max_solutions,
        finish=finish,
        tasks_completed_obj=task_completed_obj,
        tasks_completed=tasks_completed,
        total_tasks=len(early_vars),
        sum_task_slacks=sum_task_slacks,
    )

    if set_objective:
        # Maximize completed tasks
        model.Maximize(task_completed_obj)
        cb.reset()
        status = solver.Solve(model, solution_callback=cb)

        # Minimize total time and maximize slack
        for var in early_vars.values():
            model.AddHint(var.start, solver.Value(var.start))
        model.Add(tasks_completed >= solver.Value(tasks_completed))
        model.Minimize(finish - 1e3 * sum_task_slacks)
        cb.reset()
        status = solver.Solve(model, solution_callback=cb)

    else:
        status = solver.Solve(model, solution_callback=cb)

    return solver, status


def recursive_check(task: Task, visited):
    for adj in task.adjacents:
        if adj in visited:
            return adj
        else:
            res = recursive_check(adj, visited)
            visited.append(adj)
            return res


def post_process_solution(
    solver: cp_model.CpSolver,
    tasks: TaskGraph,
    start_time: datetime,
    early_vars: Dict,
    late_vars: Dict,
    presences: Dict,
    copy=False,
    fix_gtasks=False,
) -> List[Event]:
    """Set task times"""
    start_tasks = find_start_tasks(tasks)
    for task in start_tasks:
        visited = [task]
        res = recursive_check(task, visited)
        if res:
            raise ValueError(f"Recursive check failed with {res}")
    if copy:
        assert not tasks.is_cyclic()
        tasks = deepcopy(tasks)

    for task in tasks.all_tasks:
        if task.gtasks_id and fix_gtasks:
            continue
        elif task.gtasks_id and not fix_gtasks:
            task._changed = True
        if solver.Value(presences[task.task_id]) > 0:

            task.scheduled_early_start = convert_model_hours_to_datetime(
                start_time, solver.Value(early_vars[task.task_id].start)
            )
            task.scheduled_early_finish = task.scheduled_early_start + task.duration
            # task.scheduled_deadline = task.scheduled_early_start + task.duration
            task.scheduled_late_start = convert_model_hours_to_datetime(
                start_time, solver.Value(late_vars[task.task_id].start)
            )
            task.scheduled_deadline = task.scheduled_late_start + task.duration
        else:
            task.scheduled_early_start = None
            task.scheduled_early_finish = None
            task.scheduled_late_start = None
            task.scheduled_deadline = None
    return tasks


def schedule_events(
    events: List[Event],
    calendar: GoogleCalendar,
    existing_events: List[Event],
) -> List[Event]:
    for event in events:
        existing = check_existing_events(event, existing_events)
        if existing:
            existing.start = event.start
            existing.end = event.end
            calendar.update_event(existing)
        else:
            calendar.add_event(event)
    return events


def check_existing_events(
    event: Event, existing_events: List[Event]
) -> Union[None, Event]:
    for existing in existing_events:
        if event.event_id == existing.event_id:
            return existing


def optimize_schedule(
    tasks: TaskGraph,
    work_times: Dict[int, Tuple[time, time]],
    start_time: datetime,
    deadline: datetime,
    work_timezone: str = "Europe/London",
    block_duration: Optional[timedelta] = None,
) -> None:
    """

    Arguments
    ---------
    transparent_as_free: bool, optional
        Events that do not block time on the calendar are marked as available.
    """
    logger = logging.getLogger(__name__)

    n_days = (deadline - start_time).days
    filtered_availabilities = []
    timezone = pytz.timezone(work_timezone)
    for d in range(1, n_days):
        new_date = start_time + timedelta(days=d)
        work_day = work_times.get(new_date.weekday())
        if work_day:
            start_work_day = timezone.localize(
                datetime.combine(new_date.date(), work_day[0])
            )
            finish_work_day = timezone.localize(
                datetime.combine(new_date.date(), work_day[1])
            )
            filtered_availabilities.append((start_work_day, finish_work_day))
    latest_start = start_time
    latest_finish = start_time
    for a in filtered_availabilities:
        if a[0] >= latest_start:
            latest_start = a[0]
        else:
            raise ValueError(f"{a[0]} is not ordered")

        if a[1] >= latest_finish:
            latest_finish = a[1]
        else:
            raise ValueError(f"{a[1]} is not ordered")

    total_available_time = sum(
        [(a[1] - a[0]).total_seconds() / 3600 for a in filtered_availabilities]
    )
    logger.info(
        f"Total available time after filtering: {total_available_time:.01f} hours"
    )

    # Estimate if deadline is reasonable
    total_time = sum([task.duration.total_seconds() for task in tasks.all_tasks]) / 3600
    logging.info(f"Total task time: {total_time:.01f} hours")
    if total_time > total_available_time:
        raise ValueError(
            f"Total task time ({total_time}) is less than available time ({total_available_time})"
        )

    # Block duration default
    if block_duration is None:
        block_duration = timedelta(hours=1)
    if block_duration < timedelta(hours=1):
        raise ValueError("Block duration cannot be less than 1 hour")

    # Solve the internal optimization problem using pyomo
    scheduled_tasks = optimize_timing(
        tasks=tasks,
        availabilities=filtered_availabilities,
        start_time=start_time,
        deadline=deadline,
        block_duration=block_duration,
        max_solutions=50,
        timeout=120,
        separate_objectives=True,
    )

    return scheduled_tasks
