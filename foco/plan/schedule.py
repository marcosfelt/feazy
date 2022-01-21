"""
Schedule a list of tasks

## Optimization

The problem is framed in terms of the total number of work hours end to end
starting at 0 and ending at T, the total time spent on all projects in the horizon.

There are three types of tasks: intense focus, medium focus and low energy tasks. Each task has an estimated duration
and some have a deadline.

Objectives:
- Minimize total project duration 
- Maximize slack (i.e., minimizing the dependency on finishing any task in the minimum amount of time)

Decision variables:
- Early start times
- Late start times
both with bounds based on the deadlines

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
from queue import Queue
from .task import Task, TaskGraph, TaskDifficulty
from datetime import datetime, time, timedelta, date
from typing import List, Union, Tuple, Optional, Dict
from gcsa.google_calendar import GoogleCalendar, Event
from gcsa.event import Transparency
from dateutil.rrule import rrule, DAILY
from beautiful_date import *
from pyomo.environ import *
from pyomo.gdp import *
import pytz
import math
import uuid

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
    transparent_as_free: Optional[bool] = True,
    split_across_days: Optional[bool] = True,
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


def schedule_tasks(
    feazy_calendar: GoogleCalendar,
    availabilities: List[Tuple[datetime, datetime]],
    tasks: TaskGraph,
    schedule_rules: Dict[TaskDifficulty, ScheduleBlockRule],
    priority: Optional[List[TaskDifficulty]] = None,
):
    """Schedule tasks according to schedule_rules and availabilities"""
    # Get all events on the feazy calendar in the availibity time range
    feazy_events = []
    these_events = feazy_calendar.get_events(
        availabilities[0][0],
        availabilities[-1][1],
        order_by="startTime",
        single_events=True,
        showDeleted=False,
    )
    for event in these_events:
        # Exclude transparent events
        feazy_events.append(event)

    # Make sure availabilities are sorted and queued up
    availabilities.sort(key=lambda dates: dates[0])
    availability_queue = Queue()
    for availibility in availabilities:
        availability_queue.put(availibility)

    # Set default priorities
    if priority is None:
        priority = [TaskDifficulty.HARD, TaskDifficulty.MEDIUM, TaskDifficulty.EASY]

    # Schedule tasks
    task_queues = {p: Queue() for p in priority}
    all_tasks = tasks.all_tasks
    all_tasks.sort(key=lambda t: t.early_start)
    for task in all_tasks:
        task_queues[task.difficulty].put(task)
        task.total_time_spent = 0  # Time spent on task in minutes
    while (
        not (all([q.empty() for q in task_queues.values()]))
        and not availability_queue.empty()
    ):
        # For each new availability, fill with tasks in priority order
        current_availability = availability_queue.get(block=True)
        potential_events = _get_potential_blocks(current_availability, schedule_rules)
        for difficulty, task_queue in task_queues.items():
            # Get the task
            if task_queue.empty():
                continue
            elif task.total_time_spent >= task.duration:
                task = task_queue.get(block=True)

            # Get the next block if it exists
            try:
                next_events = next(potential_events[difficulty])

            except StopIteration:
                continue

            # Schedule the block
            for event in next_events:
                if event.summary not in ["Break Before", "Break After"]:

                    event.summary = task.description
                    task.total_time_spent += (
                        event.end - event.start
                    ).total_seconds() / 60
                event.event_id = uuid.uuid4().hex
                # elif event.summary == "Break Before":
                #     event.event_id = base32_crockford.normalize(
                #         "breakbefore" + task.task_id
                #     ).lower()
                # elif event.summary == "Break After":
                #     event.event_id =

            events = schedule_event(feazy_calendar, feazy_events, next_events)
            events.sort()

            # Go to to next availability if at end of current availability
            if events[-1].end >= current_availability[1]:
                break
            # Adjust availability and blocks for subsequent priority/difficulty levels
            else:
                current_availability = (events[-1].end, current_availability[1])
                potential_events = _get_potential_blocks(
                    current_availability, schedule_rules
                )
    unscheduled_count = sum(
        [q.qsize if type(q) == int else q.qsize() for q in task_queues.values()]
    )
    print("Number of unscheduled tasks:", unscheduled_count)


def _get_potential_blocks(
    availability: Tuple[datetime, datetime],
    schedule_rules: Dict[TaskDifficulty, ScheduleBlockRule],
):
    return {
        d: schedule_rule.blocks_between(availability[0], availability[1])
        for d, schedule_rule in schedule_rules.items()
    }


def schedule_event(
    calendar: GoogleCalendar, feazy_events: List[Event], block: List[Event]
) -> List[Event]:
    for event in block:
        existing = check_existing_events(event, feazy_events)
        if existing:
            existing.start = event.start
            existing.end = event.end
            calendar.update_event(existing)
        else:
            calendar.add_event(event)
    return block


def check_existing_events(
    event: Event, existing_events: List[Event]
) -> Union[None, Event]:
    for existing in existing_events:
        if event.event_id == existing.event_id:
            return existing


def create_pyomo_optimization_model(tasks: TaskGraph) -> Model:
    """Create Pyomo optimization model"""
    model = ConcreteModel()

    # Tasks is a two dimensional set of (j,m) constructed from dictionary keys
    task_keys = [(task.task_id, task.difficulty) for task in tasks.all_tasks]
    model.TASKS = Set(initialize=task_keys, dimen=2)

    # The set of jobs is constructed from a python set
    unique_tasks = list(set([j for (j, _) in model.TASKS]))
    model.JOBS = Set(initialize=unique_tasks)

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
    model.TASK_ORDER = Set(
        initialize=dependencies,
    )

    # The set of disjunctions (i.e., no time overlaps) is cross-product of jobs, jobs and LEVELS
    model.DISJUNCTIONS = Set(
        initialize=model.JOBS * model.JOBS * model.LEVELS,
        dimen=3,
        filter=lambda model, j, k, l: j < k
        and (j, l) in model.TASKS
        and (k, l) in model.TASKS,
    )

    # Load duration data into a model parameter for later acces
    model.dur = Param(model.TASKS, initialize=lambda model, j, m: tasks[j].duration)

    # Establish an upper bound on makespan
    ub = sum([model.dur[j, m] for (j, m) in model.TASKS])

    # Create decision variables
    model.makespan = Var(bounds=(0, ub))
    model.start = Var(model.TASKS, bounds=(0, ub), domain=PositiveIntegers)

    # Create ojective
    model.objective = Objective(expr=model.makespan, sense=minimize)

    # Constraint that tasks must finish before end
    model.finish = Constraint(
        model.TASKS,
        rule=lambda model, j, m: model.start[j, m] + model.dur[j, m] <= model.makespan,
    )

    # Constraint for task dependencies
    model.preceding = Constraint(
        model.TASK_ORDER,
        rule=lambda model, j, m, k, n: model.start[j, m] + model.dur[j, m]
        <= model.start[k, n],
    )

    # Constraint for non-overalapping tasks
    model.disjunctions = Disjunction(
        model.DISJUNCTIONS,
        rule=lambda model, j, k, m: [
            model.start[j, m] + model.dur[j, m] <= model.start[k, m],
            model.start[k, m] + model.dur[k, m] <= model.start[j, m],
        ],
    )

    # Transform into higher dimensional space
    TransformationFactory("gdp.hull").apply_to(model)
    return model


def schedule_solve_neos(model: Model, tasks: TaskGraph) -> None:
    # Solve
    solver_manger = SolverManagerFactory("neos")
    solver_manger.solve(model, opt="cplex")
    # SolverFactory('apopt').solve(model)

    # Update task start times
    for j, m in model.TASKS:
        tasks[j].early_start = model.start[j, m]()


def optimize_schedule(tasks: TaskGraph) -> None:
    model = create_pyomo_optimization_model(tasks)
    schedule_solve_neos(model, tasks)
