"""
Schedule a list of tasks

## Optimization

The problem is framed in terms of the total number of work hours end to end
starting at 0 and ending at T, the total time spent on all projects in the horizon.

I want to allocate tasks an early start and latest start date by looking at availability on my calendar. 
I was originally going to break down tasks into blocks, but instead I am going to let [reclaim.ai](https://reclaim.ai/)
do that. Instead, I will just assign tasks in large blocks to make sure I get reasonable early and late start dates.
Then, I will let reclaim schedule them exactly.

So the process looks like this:

Google Sheet w/ Tasks --> Optimize Early & Late Start --> Add tasks to Google tasks --> Reclaim schedules tasks

Instead of linear time, I am going to use available time in hours. The  algorithm will therefore schedule 
within this availalbe time and, then, I will need to convert back to linear time (i.e., actual dates).

The specific formulation:

Objectives:
- Minimize total project duration 
- Maximize slack (i.e., minimizing the dependency on finishing any task in the minimum amount of time)

Decision variables:
- Early start times
- Late start times
both with bounds based on the start and deadlines

Constraints:
- Tasks cannot overlap
- Task dependencies must be obeyed
- Slack must be >= 0 for all tasks

## Calendar placement
Once the exact start times are figured out, convert back to real time


"""
import pdb
from xxlimited import new

from feazy.plan.utils import GraphSearch
from .task import Task, TaskGraph
from gcsa.google_calendar import GoogleCalendar, Event
from gcsa.event import Transparency
from beautiful_date import *
from ortools.sat.python import cp_model
from datetime import datetime, time, timedelta, date
from typing import List, Union, Tuple, Optional, Dict
from queue import LifoQueue
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


def get_calendars(
    cal: GoogleCalendar, only_selected=False, exclude: Optional[List[str]] = None
) -> List[GoogleCalendar]:
    """Get all calendars associated with an account

    Arguments
    --------
    only_selected : bool
        Only get calendars that are selected in the interface.

    """
    all_calendar_dicts = cal.service.calendarList().list().execute()
    all_calendars = []
    for c in all_calendar_dicts["items"]:
        selected = c.get("selected", False)
        if only_selected and not selected:
            continue
        if c["id"] in exclude:
            continue
        all_calendars.append(GoogleCalendar(c["id"]))
    return all_calendars


def get_availability(
    calendars: Union[GoogleCalendar, List[GoogleCalendar]],
    start_time: datetime,
    end_time: datetime,
    transparent_as_free: bool = True,
    split_across_days: bool = True,
    default_timezone: Optional[str] = DEFAULT_TIMEZONE,
) -> List[Tuple[datetime, datetime]]:
    """Get availability in a particular time range

    Arguments
    ----------
    transparent_as_free : bool
        Events that do not block time on the calendar are marked as available.
    split_across_days : bool
        Break availabilities that span across days at midnight

    Returns availabilities represented as list of tuples of start and end times
    """
    logger = logging.getLogger(__name__)
    # Get events in that time range
    events = []
    if type(calendars) == GoogleCalendar:
        calendars = [calendars]
    logger.debug("Downloading events")
    for calendar in calendars:
        these_events = calendar.get_events(
            start_time,
            end_time,
            order_by="startTime",
            single_events=True,
            showDeleted=False,
        )
        for event in these_events:
            # Exclude transparent events
            if event.transparency == Transparency.TRANSPARENT and transparent_as_free:
                continue
            events.append(event)

    # Sort events by increasing time
    convert_dates_to_datetimes(events, default_timezone)
    events = _remove_duplicates(events)
    events.sort()

    # Specify availability as a list of times where there aren't events
    availabilities = []
    logger.debug("Calculating avaialabilities")
    if start_time < events[0].start:
        availability = (
            _split_across_days((start_time, events[0].start))
            if split_across_days
            else [(start_time, events[0].start)]
        )
        availabilities.extend(availability)
        latest_end = events[0].start
    else:
        latest_end = start_time
    for prev_event, next_event in zip(events, events[1:]):
        bookend = prev_event.end == next_event.start
        is_overlap = next_event.start < prev_event.end
        if prev_event.end > latest_end and not bookend and not is_overlap:
            availability = (
                _split_across_days((prev_event.end, next_event.start))
                if split_across_days
                else [(prev_event.end, next_event.start)]
            )
            availabilities.extend(availability)
        if prev_event.end > latest_end:
            latest_end = prev_event.end
    if latest_end < end_time:
        availability = (
            _split_across_days((events[-1].end, end_time))
            if split_across_days
            else [(latest_end, end_time)]
        )
        availabilities.extend(availability)
    return availabilities


def _remove_duplicates(events: List[Event]):
    new_events = events[:1]
    for event in events:
        exists = False
        for other in new_events:
            if event.start == other.start and event.end == other.end:
                exists = True
        if not exists:
            new_events.append(event)

    return new_events


def _split_across_days(
    availability: Tuple[datetime, datetime]
) -> List[Tuple[datetime, datetime]]:
    """Split an availability window at 11:59:59 each day"""

    dt = availability[1].day - availability[0].day
    availabilities = []
    if dt >= 1:
        last = availability[0]
        for _ in range(dt):
            next = last + timedelta(days=1)
            next = next.replace(hour=0, minute=0, second=0)
            availabilities.append((last, next))
            last = next
        if availability[1] > last:
            availabilities.append((last, availability[1]))
    else:
        availabilities = [availability]
    return availabilities


def convert_dates_to_datetimes(events: List[Event], default_timezone: str):
    timezone = pytz.timezone(default_timezone)
    for event in events:
        if type(event.start) == date:
            event.start = timezone.localize(
                datetime.combine(
                    event.start,
                    time(
                        0,
                        0,
                    ),
                )
            )
        if type(event.end) == date:
            event.end = timezone.localize(
                datetime.combine(event.end - timedelta(days=1), time(23, 59, 59))
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

    # Find starting tasks
    start_tasks = find_start_tasks(tasks)

    # Breakdown tasks
    visit_queue = LifoQueue()  # LIFO queue (stack) for doing depth first search
    visited_tasks = []
    for start_task in start_tasks:
        for adj in start_task.adjacents:
            visit_queue.put(adj)

        while not visit_queue.empty():
            current_task = visit_queue.get(block=True)
            if current_task not in visited_tasks:
                # Break up task if necessary
                if current_task.duration > block_duration:
                    n_blocks = int(math.ceil(current_task.duration / block_duration))
                    new_tasks[current_task.task_id].duration = block_duration
                    new_tasks[
                        current_task.task_id
                    ].description += f" (Block 1/{n_blocks})"
                    last_block = new_tasks[current_task.task_id]
                    for b in range(n_blocks - 1):
                        new_block = Task(
                            task_id=current_task.task_id + f"_{b+2}",
                            duration=block_duration,
                            description=current_task.description
                            + f" (Block {b+2}/{n_blocks})",
                        )
                        new_tasks.add_task(new_block)
                        new_tasks.add_dependency(last_block.task_id, new_block.task_id)
                        last_block = new_block
                    for adj in current_task.adjacents:
                        new_tasks.remove_dependency(current_task.task_id, adj.task_id)
                        new_tasks.add_dependency(last_block.task_id, adj.task_id)
                else:
                    last_block = new_tasks[current_task.task_id]

                # Add new adjacents
                for adj in current_task.adjacents:
                    if adj not in visited_tasks:
                        visit_queue.put(adj, block=True)

                # Mark current node as visited
                visited_tasks.append(current_task)

    return new_tasks


def consolidate_tasks(original_tasks: TaskGraph, block_tasks: TaskGraph) -> TaskGraph:
    """Consolidate tasks after breadking them down"""
    new_tasks = deepcopy(original_tasks)

    # Consolidate tasks
    for current_task in block_tasks.all_tasks:
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
        raise_none(new_tasks[id].scheduled_early_start)

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
    (
        model,
        early_vars,
        late_vars,
        late_finish_obj_var,
        sum_task_slacks,
    ) = create_optimization_model_time_based(
        task_blocks, availabilities, start_time, deadline, set_objective=False
    )
    solver, status = solve_cpsat(
        model,
        early_vars=early_vars,
        late_vars=late_vars,
        max_solutions=max_solutions,
        late_finish=late_finish_obj_var,
        sum_task_slacks=sum_task_slacks,
    )

    # Update tasks with scheduled start times and deadlines
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        new_tasks = post_process_solution(
            solver, task_blocks, start_time, early_vars, late_vars
        )

        # Consolidate tasks
        consolidated_tasks = consolidate_tasks(
            original_tasks=tasks, block_tasks=new_tasks
        )
        return consolidated_tasks
    else:
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
) -> Tuple[cp_model.CpModel, Dict, Dict]:
    """Create OR-tools optimization model. Time is in hours"""
    logger = logging.getLogger(__name__)
    # Create model
    model = cp_model.CpModel()

    # Create variables for early and late starts + intervals
    early_vars = {}
    late_vars = {}
    project_deadline_hours = convert_datetime_to_model_hours(start_time, deadline)
    logger.debug(f"Project deadline hours: {project_deadline_hours}")
    for task in tasks.all_tasks:
        for name, var_group in {
            "early": early_vars,
            "late": late_vars,
        }.items():
            lb = (
                0
                if task.earliest_start is None
                else convert_datetime_to_model_hours(start_time, task.earliest_start)
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
            start_hint = (
                task.scheduled_early_start
                if name == "early"
                else task.scheduled_late_start
            )
            # if start_hint:
            #     hint = convert_datetime_to_model_hours(start_time, start_hint)
            #     model.AddHint(start_var, hint)
            dur = int(task.duration.total_seconds() / 3600)
            interval_var = model.NewFixedSizeIntervalVar(
                start_var, dur, f"{name}_interval_{task.task_id}"
            )
            var_group[task.task_id] = task_type(
                start=start_var, end=start_var + dur, interval=interval_var
            )

    # Hint at initial values
    # start_tasks = find_start_tasks(tasks)
    # for start_task in start_tasks:
    #     current_time = 0
    #     for task in tasks.graph_search(start_task.task_id, GraphSearch.DFS):
    #         # Move to earliest start if need to
    #         lb = (
    #             0
    #             if task.earliest_start is None
    #             else convert_datetime_to_model_hours(start_time, task.earliest_start)
    #         )
    #         if lb > current_time:
    #             current_time = lb

    #         # Add hint
    #         model.AddHint(early_vars[task.task_id].start, current_time)
    #         model.AddHint(late_vars[task.task_id].start, current_time + 72)

    #         # Increment naively
    #         dur = int(task.duration.total_seconds() / 3600)
    #         wait_time = int(task.wait_time.total_seconds() / 3600)
    #         current_time += dur + wait_time

    # Define non-available intervals
    availabilities.sort(key=lambda a: a[0], reverse=False)
    busy_intervals = []
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
            busy_intervals.append(
                model.NewFixedSizeIntervalVar(start, size, f"busy_interval_{i}")
            )

    # No overlap constraint with availability constraints
    for var_group in [early_vars, late_vars]:
        model.AddNoOverlap(
            [group.interval for group in var_group.values()] + busy_intervals[30:]
        )

    # model.AddDecisionStrategy(
    #     [t.start for t in early_vars.values()],
    #     cp_model.CHOOSE_FIRST,
    #     cp_model.SELECT_MIN_VALUE,
    # )

    # Precedence constraint with wait times
    for task in tasks.all_tasks:
        for var_group in [early_vars, late_vars]:
            for succ in task.successors:
                wait_time = int(task.wait_time.total_seconds() / 3600)
                model.Add(
                    var_group[task.task_id].end + wait_time
                    <= var_group[succ.task_id].start
                )

    # Slack constraint (early start must before late start)
    for task in tasks.all_tasks:
        model.Add(early_vars[task.task_id].start <= late_vars[task.task_id].start)

    # Objectives: minimize late finish while maximizing slack
    sum_task_slacks = sum(
        [
            late_vars[task.task_id].start - early_vars[task.task_id].start
            for task in tasks.all_tasks
        ]
    )
    late_finish_obj_var = model.NewIntVar(
        0, project_deadline_hours * len(early_vars), "late_finish"
    )
    model.AddMaxEquality(
        late_finish_obj_var,
        [late_vars[task.task_id].end for task in tasks.all_tasks],
    )
    if set_objective:
        model.Minimize(late_finish_obj_var - sum_task_slacks)

    return model, early_vars, late_vars, late_finish_obj_var, sum_task_slacks


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, max_solutions: int, late_finish, sum_task_slacks):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._solution_count = 0
        self._solution_limit = max_solutions
        self._late_finish = late_finish
        self._sum_task_slacks = sum_task_slacks

    def OnSolutionCallback(self):
        return self.on_solution_callback()

    def on_solution_callback(self):
        self._solution_count += 1

        late_finish = self.Value(self._late_finish)
        sum_task_slacks = self.Value(self._sum_task_slacks)

        print(f"Late finish: {late_finish} | Task Slacks: {sum_task_slacks}")

        if self._solution_count >= self._solution_limit:
            print(f"Stop search after {self._solution_count} solutions.")
            self.StopSearch()

    @property
    def solution_count(self):
        return self._solution_count


def solve_cpsat(
    model: cp_model.CpModel,
    late_finish: cp_model.IntVar,
    sum_task_slacks: cp_model.IntVar,
    early_vars: Dict[str, task_type],
    late_vars: Dict[str, task_type],
    max_solutions=10,
    num_workers=6,
    log_search_progress=True,
) -> None:
    """Solve optimization problem using CpSAT"""
    logger = logging.getLogger(__name__)
    logger.debug("Solving optimization problem using CPSat")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = log_search_progress
    solver.parameters.num_search_workers = num_workers

    # First minimize late finish
    model.Minimize(late_finish)
    cb = SolutionCallback(max_solutions, late_finish, sum_task_slacks)
    status = solver.Solve(model, solution_callback=cb)

    # Hint/constraint values of variables based on first objective
    for var_group in [early_vars, late_vars]:
        for var in var_group.values():
            model.AddHint(var.start, solver.Value(var.start))
    model.Add(late_finish == round(solver.ObjectiveValue()))

    # Then maximize task slacks
    model.Maximize(sum_task_slacks)
    cb = SolutionCallback(max_solutions, late_finish, sum_task_slacks)
    status = solver.Solve(model, solution_callback=cb)

    return solver, status


def post_process_solution(
    solver: cp_model.CpSolver,
    tasks: TaskGraph,
    start_time: datetime,
    early_vars: Dict,
    late_vars: Dict,
    copy=True,
) -> List[Event]:
    """Set task times"""
    if copy:
        tasks = deepcopy(tasks)

    for task in tasks.all_tasks:
        task.scheduled_early_start = convert_model_hours_to_datetime(
            start_time, solver.Value(early_vars[task.task_id].start)
        )
        task.scheduled_early_finish = task.scheduled_early_start + task.duration
        task.scheduled_late_start = convert_model_hours_to_datetime(
            start_time, solver.Value(late_vars[task.task_id].start)
        )
        task.scheduled_deadline = task.scheduled_late_start + task.duration
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
    base_calendar: GoogleCalendar,
    work_timezone: str = "Europe/London",
    block_duration: Optional[timedelta] = None,
    exclude_calendar_ids: Optional[List[str]] = None,
    transparent_as_free: bool = True,
) -> None:
    """

    Arguments
    ---------
    transparent_as_free: bool, optional
        Events that do not block time on the calendar are marked as available.
    """
    logger = logging.getLogger(__name__)

    # # Get availability
    # exclude_calendar_ids = (
    #     exclude_calendar_ids if exclude_calendar_ids is not None else []
    # )
    # exclude_calendar_ids = list(set(exclude_calendar_ids))
    # calendars = get_calendars(base_calendar, exclude=exclude_calendar_ids)
    # availabilities = get_availability(
    #     calendars,
    #     start_time,
    #     deadline,
    #     transparent_as_free=transparent_as_free,
    #     split_across_days=False,
    # )
    # total_available_time = sum(
    #     [(a[1] - a[0]).total_seconds() / 3600 for a in availabilities]
    # )
    # logger.info(
    #     f"Total available time before filtering: {total_available_time:.01f} hours"
    # )
    # # for a in availabilities:
    # #     print(fmt_date(a[0]), "-", fmt_date(a[1]))

    # # Filter availabilities to work times
    # filtered_availabilities = filter_availabilities(availabilities, work_times)
    # filtered_availabilities.sort(key=lambda d: d[0], reverse=False)

    n_days = (deadline - start_time).days
    filtered_availabilities = []
    timezone = pytz.timezone(work_timezone)
    for d in range(n_days):
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
        max_solutions=2,
    )

    return scheduled_tasks
