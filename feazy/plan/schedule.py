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

from .task import Task, TaskGraph, TaskDifficulty
from .utils import GraphSearch
from gcsa.google_calendar import GoogleCalendar, Event
from gcsa.event import Transparency
from dateutil.rrule import rrule, DAILY
from beautiful_date import *
from pyomo.environ import *
from pyomo.gdp import *
from datetime import datetime, time, timedelta, date
from typing import List, Union, Tuple, Optional, Dict
from queue import LifoQueue
import pytz
import math
import numpy as np
import pandas as pd
from copy import deepcopy

fmt_date = lambda d: d.astimezone(pytz.timezone("UTC")).strftime("%m/%d/%Y, %H:%M:%S")


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
    # Get events in that time range
    events = []
    if type(calendars) == GoogleCalendar:
        calendars = [calendars]
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
    events.sort()

    # Specify availability as a list of times where there aren't events
    availabilities = []
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
    if events[-1].end < end_time:
        availability = (
            _split_across_days((events[-1].end, end_time))
            if split_across_days
            else [(events[-1].end, end_time)]
        )
        availabilities.extend(availability)

    return availabilities


def _split_across_days(
    availability: Tuple[datetime, datetime]
) -> List[Tuple[datetime, datetime]]:
    """Split an availability window at midnight each day"""

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


class ScheduleBlockRule(rrule):
    """Rules for blocks on schedules. Only daily for now

    Arguments
    ---------
    block_duration : int
        The amount of time in minutes in a single block
    max_blocks: int, optional
        The maximum number of blocks in a period. So if `max_blocks=2`
        and `freq=WEEKLY`, a maximum two blocks can be scheduled per week.
        Defaults to 1.
    earliest_time : datetime.time, optional
        The earliest time a block can be scheduled in a given day. Defaults to 9 AM UTC.
    latest_time : datetime.time, optional
        The latest time a block can be scheduled in a given day. Defaults to 7 PM UTC.
    break_duration_before : int, optional
        Duration of break before block in mintues. Defaults to 0 (i.e., no break)
    break_duration : int, optional
        Duration of break after block in mintues. Defaults to 0 (i.e., no break)
    **kwargs
        See https://dateutil.readthedocs.io/en/stable/rrule.html#classes
    """

    def __init__(
        self,
        block_duration: int,
        max_blocks: Optional[int] = 1,
        earliest_time: Optional[time] = None,
        latest_time: Optional[time] = None,
        break_duration_before: Optional[int] = 0,
        break_duration_after: Optional[int] = 0,
        **kwargs,
    ) -> None:
        self.block_duration = block_duration
        self.max_blocks = max_blocks
        self.break_duration_before = break_duration_before
        self.break_duration_after = break_duration_after
        timezone = pytz.timezone("UTC")
        self._earliest_time = (
            timezone.localize(time(hour=9, minute=0))
            if earliest_time is None
            else earliest_time
        )
        self._latest_time = (
            timezone.localize(time(hour=17, minute=0))
            if latest_time is None
            else latest_time
        )
        self._block_count = {}
        # Cache for speed
        if kwargs.get("cache", None):
            kwargs["cache"] = True
        super().__init__(freq=DAILY, **kwargs)

    @property
    def earliest_time(self) -> time:
        return self._earliest_time

    @earliest_time.setter
    def earliest_time(self, t: time):
        if self.latest_time is not None:
            assert t < self.latest_time
        self._earliest_time = t

    @property
    def latest_time(self) -> time:
        return self._latest_time

    @latest_time.setter
    def latest_time(self, t: time):
        if self.earliest_time is not None:
            assert t > self.earliest_time
        self._latest_time = t

    def blocks_between(
        self, start_time: datetime, end_time: datetime
    ) -> List[Optional[Event]]:

        # Check if times within early/late times
        earliest_time = datetime.combine(
            start_time.date(), self.earliest_time, tzinfo=start_time.tzinfo
        )
        if start_time < earliest_time:
            start_time = earliest_time

        if start_time > end_time:
            return []
        latest_time = datetime.combine(
            start_time.date(), self.latest_time, tzinfo=start_time.tzinfo
        )
        if end_time > latest_time:
            end_time = latest_time

        # Check for recurrence rule
        item = self.before(start_time.replace(tzinfo=None), inc=True)
        check_rule = False
        if item:
            if item.date() == start_time.date():
                check_rule = True

        # Check for max block counts
        if not self._block_count.get(start_time.date()):
            self._block_count[start_time.date()] = 0
        check_count = self._block_count[start_time.date()] < self.max_blocks

        if check_rule and check_count:
            # Total time in minutes
            dt = (end_time - start_time).total_seconds() / 60
            # Block time
            block_and_break = (
                self.block_duration
                + self.break_duration_before
                + self.break_duration_after
            )
            # Always round down for number of blocks
            num_blocks = math.floor(dt / block_and_break)

            self.last_end = start_time
            for _ in range(num_blocks):
                events = []
                # Break before
                if self.break_duration_before > 0:
                    end = self.last_end + timedelta(minutes=self.break_duration_before)
                    events.append(
                        Event(start=self.last_end, end=end, summary="Break Before")
                    )
                    self.last_end = end

                # Block
                end = self.last_end + timedelta(minutes=self.block_duration)
                events.append(Event(start=self.last_end, end=end, summary=""))
                self.last_end = end

                # Break after
                if self.break_duration_after > 0:
                    end = self.last_end + timedelta(minutes=self.break_duration_after)
                    events.append(
                        Event(start=self.last_end, end=end, summary="Break After")
                    )
                    self.last_end = end
                yield events
                self._block_count[start_time.date()] += 1
        else:
            return []


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


def breakdown_tasks(
    tasks: TaskGraph, schedule_rules: Dict[TaskDifficulty, ScheduleBlockRule]
) -> TaskGraph:
    """Breakdown tasks into smaller tasks + breaks according to schedule rules"""
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
                schedule_rule = schedule_rules[current_task.difficulty]

                # Insert break before
                break_before = schedule_rule.break_duration_before
                if break_before > 0:
                    new_break = Task(
                        TaskDifficulty.EASY,
                        duration=break_before,
                        description="Break before next task",
                    )
                    new_tasks.add_task(new_break)
                    predecessors_ids = []
                    for t in new_tasks.graph_search(
                        start_task.task_id, type=GraphSearch.BFS
                    ):
                        if current_task in t.adjacents:
                            predecessors_ids.append(t.task_id)
                        elif t == current_task:
                            break

                    for predecessor_id in predecessors_ids:
                        new_tasks.remove_dependency(
                            predecessor_id, current_task.task_id
                        )
                        new_tasks.add_dependency(predecessor_id, new_break.task_id)
                    new_tasks.add_dependency(new_break.task_id, current_task.task_id)

                # Break up task if necessary
                if current_task.duration > schedule_rule.block_duration:
                    n_blocks = math.ceil(
                        schedule_rule.block_duration / current_task.duration
                    )
                    new_tasks[
                        current_task.task_id
                    ].duration = schedule_rule.block_duration
                    new_tasks[
                        current_task.task_id
                    ].description += f" (Block 1/{n_blocks})"
                    last_block = new_tasks[current_task.task_id]
                    for b in range(n_blocks - 1):
                        new_block = Task(
                            current_task.task_id,
                            duration=schedule_rule.block_duration,
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

                # Insert break after
                break_after = schedule_rule.break_duration_after
                if break_after > 0:
                    new_break = Task(
                        TaskDifficulty.EASY,
                        duration=break_before,
                        description="Break after previous task",
                    )
                    new_tasks.add_task(new_break)
                    for adj in last_block.adjacents:
                        new_tasks.remove_dependency(last_block.task_id, adj.task_id)
                        new_tasks.add_dependency(new_break.task_id, adj.task_id)

                # Add new adjacents
                for adj in current_task.adjacents:
                    if adj not in visited_tasks:
                        visit_queue.put(adj, block=True)

                # Mark current node as visited
                visited_tasks.append(current_task)

    return new_tasks


def _optimize_timing(
    tasks: TaskGraph,
    availabilities: List[Tuple[datetime, datetime]],
    work_times: Dict[int, Dict],
) -> TaskGraph:
    # Filter availabilities to work times
    filtered_availabilities = filter_availabilities(availabilities, work_times)

    # Convert availabilities
    total_hours, conversion_table = convert_availabilities_to_hours(
        filtered_availabilities
    )

    # Formulate and solve optimizaton problem
    model = create_pyomo_optimization_model(tasks, total_hours, conversion_table)
    schedule_solve_neos(model)

    # Convert solved model to events
    new_tasks = post_process_model(model, tasks, conversion_table)
    return new_tasks


def filter_availabilities(
    availabilities: List[Tuple[datetime, datetime]], work_times: Dict[int, Dict]
) -> List[Tuple[datetime, datetime]]:
    """Filter availabilities to work times"""
    pass


def convert_availabilities_to_hours(
    availabilities: List[Tuple[datetime, datetime]]
) -> Tuple[int, pd.DataFrame]:
    """Convert availabilities to a total hours for project"""
    data = []
    total_hours = 0
    availabilities.sort(key=lambda d: d[0], reverse=False)
    for availability in availabilities:
        duration = (availability[1] - availability[0]).total_seconds() / 3600
        data.append(
            {
                "start_datetime": availability[0],
                "end_datetime": availability[1],
                "start_hour": total_hours,
                "end_hour": total_hours + duration,
            }
        )
        total_hours += duration
    df = pd.DataFrame(data)
    for col in ["start_datetime", "end_datetime"]:
        df[col] = pd.to_datetime(df[col])
    return total_hours, df


def convert_time_to_hours(time: datetime, conversion_table: pd.DataFrame):
    row = conversion_table[
        (conversion_table["start_datetime"] >= time)
        & (conversion_table["end_datetime"] <= time)
    ]
    return row["start_hour"]


def convert_hours_to_time(hour: int, conversion_table: pd.DataFrame):
    row = conversion_table[
        (conversion_table["start_hour"] >= hour)
        & (conversion_table["end_hour"] <= hour)
    ]
    return row["start_datetime"]


def create_pyomo_optimization_model(
    tasks: TaskGraph, deadline_hours: int, conversion_table: pd.DataFrame
) -> Model:
    """Create Pyomo optimization model"""

    model = ConcreteModel()

    #### Setup ####
    # Tasks is a set of task ids
    task_keys = [task.task_id for task in tasks.all_tasks]
    model.TASKS = Set(initialize=task_keys, dimen=1)

    # The order of tasks is based on the adjacent tasks
    dependencies = list(
        set(
            [
                (task.task_id, adj.task_id)
                for task in tasks.all_tasks
                for adj in task.adjacents
            ]
        )
    )
    model.TASK_ORDER = Set(initialize=dependencies)

    # The set of disjunctions (i.e., no time overlaps) is cross-product of all tasks
    model.DISJUNCTIONS = Set(
        initialize=model.TASKS * model.TASKS,
        dimen=2,
        filter=lambda model, t, u: t < u and t in model.TASKS and u in model.TASKS,
    )

    # Load duration data into a model parameter
    model.dur = Param(model.TASKS, initialize=lambda model, t: tasks[t].duration)

    #### Decision Variables ####
    lb = [  # Bound by 0 or specified earliest start for tasks
        0
        if task.earliest_start is None
        else convert_time_to_hours(task.earliest_start, conversion_table)
        for task in tasks.all_tasks
    ]
    ub = [
        deadline_hours
        if task.deadline is None
        else convert_time_to_hours(task.deadline, conversion_table)
        for task in tasks.all_tasks
    ]
    model.early_start = Var(model.TASKS, bounds=(lb, ub), domain=PositiveIntegers)
    model.late_start = Var(model.TASKS, bounds=(lb, ub), domain=PositiveIntegers)

    #### Objectives ####
    # Maximize slack
    total_task_slack = sum(
        [model.late_start[t] - model.early_start[t] for t in model.TASKS]
    )
    model.slack = Var(bounds=(0, total_task_slack))
    model.latest_finish = Var(bounds=(0, deadline_hours), domain=PositiveIntegers)
    model.objective = Objective(expr=model.slack - model.latest_finish, sense=maximize)

    #### Constraints ####
    # Constraint that late finish must be before latest finish (which is in turn before deadline)
    model.finish = Constraint(
        model.TASKS,
        rule=lambda model, t: model.late_start[t] + model.dur[t] <= model.latest_finish,
    )

    # Constraint for task dependencies
    model.preceding_early_start = Constraint(
        model.TASK_ORDER,
        rule=lambda model, t, u: model.early_start[t] + model.dur[t]
        <= model.early_start[u],
    )
    model.preceding_late_start = Constraint(
        model.TASK_ORDER,
        rule=lambda model, t, u: model.late_start[t] + model.dur[t]
        <= model.late_start[u],
    )

    # Constraint for non-overalapping tasks (might change later)
    model.disjunctions_early_start = Disjunction(
        model.DISJUNCTIONS,
        rule=lambda model, t, u: [
            model.early_start[t] + model.dur[t] <= model.early_start[u],
            model.early_start[u] + model.dur[u] <= model.early_start[t],
        ],
    )
    model.disjunctions_late_start = Disjunction(
        model.DISJUNCTIONS,
        rule=lambda model, t, u: [
            model.late_start[t] + model.dur[t] <= model.late_start[u],
            model.late_start[u] + model.dur[u] <= model.late_start[t],
        ],
    )

    # Make sure late start is always after early start
    model.slacks = Constraint(
        model.TASKS,
        rule=lambda model, t: model.late_start[t] - model.early_start[t] >= 0,
    )

    # Transform into higher dimensional space
    TransformationFactory("gdp.hull").apply_to(model)
    return model


def post_process_model(
    model: ConcreteModel, tasks: TaskGraph, conversion_table: pd.DataFrame, copy=True
) -> List[Event]:
    """Set early start and late start of tasks using conversion table"""
    if copy:
        tasks = deepcopy(tasks)
    for task in tasks.all_tasks:
        task.early_start = convert_hours_to_time(
            model.early_start[task.task_id], conversion_table
        )
        task.late_start = convert_hours_to_time(
            model.late_start[task.task_id], conversion_table
        )
    return tasks


def schedule_solve_neos(model: Model) -> None:
    # Solve
    solver_manger = SolverManagerFactory("neos")
    solver_manger.solve(model, opt="cplex")
    # SolverFactory('apopt').solve(model)


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
    schedule_rules: Dict[TaskDifficulty, ScheduleBlockRule],
    start_time: datetime,
    end_time: datetime,
    base_calendar: GoogleCalendar,
    exclude_calendar_ids: Optional[List[str]] = None,
    transparent_as_free: bool = True,
) -> None:
    """

    Arguments
    ---------
    transparent_as_free: bool, optional
        Events that do not block time on the calendar are marked as available.
    """
    # Get availability
    exclude_calendar_ids = (
        exclude_calendar_ids if exclude_calendar_ids is not None else []
    )
    exclude_calendar_ids = list(set(exclude_calendar_ids))
    calendars = get_calendars(base_calendar, exclude=exclude_calendar_ids)
    availabilities = get_availability(
        calendars, start_time, end_time, transparent_as_free=transparent_as_free
    )

    # Solve the internal optimization problem using pyomo
    new_tasks = _optimize_timing(
        tasks=tasks, availabilities=availabilities, schedule_rules=schedule_rules
    )

    return new_tasks
