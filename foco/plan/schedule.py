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
from .task import TaskGraph
from datetime import datetime
from typing import List, Union, Tuple, Optional
from gcsa.google_calendar import GoogleCalendar, Event
from beautiful_date import *
from pyomo.environ import *
from pyomo.gdp import *
import pytz


fmt_date = lambda d: d.astimezone(pytz.timezone("UTC")).strftime("%m/%d/%Y, %H:%M:%S")


class Rules:
    pass


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


def handle_recurrence(event: Event, earliest_time: datetime) -> None:
    if event.recurrence and event.start < earliest_time:
        start_time = datetime(
            earliest_time.year,
            earliest_time.month,
            earliest_time.day,
            event.start.hour,
            event.start.minute,
            tzinfo=event.start.tzinfo,
        )
        end_time = datetime(
            earliest_time.year,
            earliest_time.month,
            earliest_time.day,
            event.end.hour,
            event.end.minute,
            tzinfo=event.end.tzinfo,
        )
        event.start = start_time
        event.end = end_time


def get_availability(
    calendars: Union[GoogleCalendar, List[GoogleCalendar]],
    start_time: datetime,
    end_time: datetime,
) -> List[Tuple[datetime, datetime]]:
    """Get availability in a particular time range

    Ideally focus on one day for this to work best

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
            # order_by="startTime",
            # single_events=False,
        )
        for event in these_events:
            handle_recurrence(event, earliest_time=start_time)
            events.append(event)

    # Sort events by time
    events.sort()
    for event in events:
        print(event)

    # Specify availability as a list of times where there aren't events
    availabilities = []
    latest_end = start_time
    for prev_event, next_event in zip(events, events[1:]):
        bookend = prev_event.end == next_event.start
        if prev_event.end > latest_end and not bookend:
            availabilities.append((prev_event.end, next_event.start))
            latest_end = event.end

    return availabilities


def schedule_tasks(
    calendar, tasks: TaskGraph, start_time: datetime, schedule_rules: Rules
):
    """Schedule tasks starting from start_time according to schedule_rules"""

    pass


def create_pyomo_optimization_model(tasks: TaskGraph) -> Model:
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


# def formulate_optimization(
#     tasks: TaskGraph,
#     dependencies: Graph,
# ):
#     pass
#     # Specify decision variables for early and late start times for tasks


#     ### Constraints

#     # Specify tasks cannot overlap

#     # Specify task dependencies

#     # Slack must be greater than 0 all tasks

#     # Tasks must have to be happen deadlines

#     ### Set objective

#     # Calculate total time

#     # Calculate total slack

#     ### Translate optimization results back to tasks
