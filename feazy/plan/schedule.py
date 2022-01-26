"""
Schedule a list of tasks

## Optimization

The problem is framed in terms of the total number of work hours end to end
starting at 0 and ending at T, the total time spent on all projects in the horizon.

There are three types of tasks: intense focus, medium focus and low energy tasks. Each task has an estimated duration
and some have a start or deadline date.

The idea of my formulation is to think of 

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


- 2 hours of intense focus per day
- 2 hours of medium focus per day
- 4 hours of low energy focus per day


Solve optimization problem
min(project_duration-total_slack)
# s.t. tau_i

"""

from .task import Task, TaskGraph, TaskDifficulty
from gcsa.google_calendar import GoogleCalendar, Event
from gcsa.event import Transparency
from dateutil.rrule import rrule, DAILY
from beautiful_date import *
from pyomo.environ import *
from pyomo.gdp import *
from datetime import datetime, time, timedelta, date
from typing import List, Union, Tuple, Optional, Dict
from queue import Queue
import pytz
import math

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


def breakdown_tasks(
    tasks: TaskGraph, schedule_rules: Dict[TaskDifficulty, ScheduleBlockRule]
) -> TaskGraph:
    """Breakdown tasks into smaller tasks + breaks according to schedule rules"""
    pass


def _optimize_schedule(
    tasks: TaskGraph,
    availabilities: List[Tuple[datetime, datetime]],
    schedule_rules: Dict[TaskDifficulty, ScheduleBlockRule],
) -> List[Event]:
    # Formulate and solve optimizaton problem
    model = create_pyomo_optimization_model(tasks, availabilities, schedule_rules)
    schedule_solve_neos(model)

    # Convert solved model to events
    events = post_process_model(model, tasks, availabilities, schedule_rules)
    return events


convert_time_to_minutes = lambda start, time: (time - start).total_seconds / 60


def create_pyomo_optimization_model(
    start_time: datetime,
    tasks: TaskGraph,
    existing_events: List[Event],
    schedule_rules: Dict[TaskDifficulty, ScheduleBlockRule],
) -> Model:
    """Create Pyomo optimization model"""

    model = ConcreteModel()

    #### Setup ####
    # Tasks is a two dimensional set of (j,m) constructed from dictionary keys
    task_keys = [(task.task_id, task.difficulty) for task in tasks.all_tasks]
    model.TASKS = Set(initialize=task_keys, dimen=2)

    # The set of events is constructed from a python set
    unique_tasks = list(set([e for (e, _) in model.TASKS]))
    model.EVENTS = Set(initialize=unique_tasks)

    # Set of LEVELS is constructed from a python set
    unique_LEVELS = list(set([l for (_, l) in model.TASKS]))
    model.LEVELS = Set(initialize=unique_LEVELS)

    # The order of tasks is based on the adjacent tasks
    dependencies = list(
        set(
            [
                (task.task_id, task.difficulty, adj.task_id, adj.difficulty)
                for task in tasks.all_tasks
                for adj in task.adjacents
            ]
        )
    )
    model.TASK_ORDER = Set(initialize=dependencies)

    # The set of disjunctions (i.e., no time overlaps) is cross-product of jobs, jobs and LEVELS
    model.DISJUNCTIONS = Set(
        initialize=model.JOBS * model.JOBS * model.LEVELS,
        dimen=3,
        filter=lambda model, j, k, l: j < k
        and (j, l) in model.TASKS
        and (k, l) in model.TASKS,
    )

    # Load duration data into a model parameter
    model.dur = Param(model.TASKS, initialize=lambda model, j, m: tasks[j].duration)

    # Total time
    total_time = sum([model.dur[j, m] for (j, m) in model.TASKS])

    #### Decision Variables ####
    lb = [  # Bound by 0 or specified earliest start for tasks
        0
        if task.earliest_start is None
        else convert_time_to_minutes(task.earliest_start)
        for task in tasks.all_tasks
    ]
    ub = [
        total_time if task.deadline is None else convert_time_to_minutes(task.deadline)
        for task in tasks.all_tasks
    ]
    model.early_start = Var(model.TASKS, bounds=(lb, ub), domain=PositiveIntegers)
    model.late_start = Var(model.TASKS, bounds=(lb, ub), domain=PositiveIntegers)

    # Calculate slack
    total_slack = sum(
        [model.late_start[e, t] - model.early_start[e, t] for (e, t) in model.TASKS]
    )

    #### Objectives ####
    model.early_finish = Var(bounds=(0, total_time))
    model.slack = Var(bounds=(0, total_slack))
    model.objective = Objective(expr=model.early_finish - model.slack, sense=minimize)

    #### Constraints ####
    # Constraint that tasks must finish before end
    model.finish = Constraint(
        model.TASKS,
        rule=lambda model, j, m: model.early_start[j, m] + model.dur[j, m]
        <= model.early_finish,
    )

    # Constraint for task dependencies
    model.preceding_early_start = Constraint(
        model.TASK_ORDER,
        rule=lambda model, j, m, k, n: model.early_start[j, m] + model.dur[j, m]
        <= model.early_start[k, n],
    )
    model.preceding_late_start = Constraint(
        model.TASK_ORDER,
        rule=lambda model, j, m, k, n: model.late_start[j, m] + model.dur[j, m]
        <= model.late_start[k, n],
    )

    # Constraint for non-overalapping tasks
    model.disjunctions_early_start = Disjunction(
        model.DISJUNCTIONS,
        rule=lambda model, e, f, l: [
            model.early_start[e, l] + model.dur[e, l] <= model.early_start[f, l],
            model.early_start[f, l] + model.dur[f, l] <= model.early_start[e, l],
        ],
    )
    model.disjunctions_late_start = Disjunction(
        model.DISJUNCTIONS,
        rule=lambda model, e, f, l: [
            model.late_start[e, l] + model.dur[e, l] <= model.late_start[f, l],
            model.late_start[f, l] + model.dur[f, l] <= model.late_start[e, l],
        ],
    )

    # Slack constraint
    model.slacks = Constraint(
        model.TASKS,
        rule=lambda model, j, m: model.late_start[j, m] - model.early_start[j, m] >= 0,
    )

    # Existing event constraint
    model.existing_events = ConstraintList()
    for st in [model.early_start, model.late_start]:
        for event in existing_events:
            model.existing_events.add(
                model.TASKS,
                rule=lambda model, t, d: st[t, d]
                < convert_time_to_minutes(start_time, event.start),
            )
            model.existing_events.add(
                model.TASKS,
                rule=lambda model, t, d: st[t, d] + model.dur[t, d]
                > convert_time_to_minutes(start_time, event.end),
            )

    # Schedule rules
    for difficulty, rule in schedule_rules.items():
        pass

    # Transform into higher dimensional space
    TransformationFactory("gdp.hull").apply_to(model)
    return model


def post_process_model(
    model: ConcreteModel,
    tasks: TaskGraph,
    availabilities: List[Tuple[datetime, datetime]],
    schedule_rules: Dict[TaskDifficulty, ScheduleBlockRule],
) -> List[Event]:
    pass


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
    feazy_calendar: GoogleCalendar,
    exclude_calendar_ids: Optional[List[str]] = None,
    transparent_as_free: bool = True,
) -> None:
    """

    Arguments
    ---------
    transparent_as_free: bool, optional
        Events that do not block time on the calendar are marked as available.
    """
    # Breakdown tasks
    blocks = breakdown_tasks(tasks, schedule_rules)

    # Get availability
    calendars = get_calendars(base_calendar, exclude=exclude_calendar_ids)
    availabilities = get_availability(
        calendars, start_time, end_time, transparent_as_free=transparent_as_free
    )

    # Solve the internal optimization problem using pyomo
    new_events = _optimize_schedule(
        tasks == blocks, availabilities=availabilities, schedule_rules=schedule_rules
    )

    # Schedule tasks
    feazy_events = feazy_calendar.get_events(
        availabilities[0][0],
        availabilities[-1][1],
        order_by="startTime",
        single_events=True,
        showDeleted=False,
    )
    schedule_events(
        events=new_events, calendar=feazy_calendar, existing_events=feazy_events
    )
