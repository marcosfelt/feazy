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
from .task import Task, TaskGraph
from .utils import Graph
from typing import List, Dict
import pandas as pd
from datetime import date, datetime
from pyomo.environ import *
from pyomo.gdp import *

from foco.plan import task


class Rules:
    pass


class CalendarItem:
    pass


def generate_time_forecast(
    current_schedule: List[CalendarItem],
    work_rules: Rules,
    earliest_task: int,
    latest_task: int,
) -> pd.DataFrame:
    # Generate an available time forecast
    # taking into consideration current schedule, work rules and working hours.
    pass


class TaskTimeConverter:
    def __init__(self, time_forecast: pd.DataFrame, level_names: List[str]):
        self.time_forecast = time_forecast
        self.level_names = level_names

    def task_to_date(self, task_time: int, task_level: str):
        pass

    def date_to_task(self, date: datetime, task_level: str):
        pass


TASKS = {
    ("Paper_1", "Blue"): {"dur": 45, "prec": None},
    ("Paper_1", "Yellow"): {"dur": 10, "prec": ("Paper_1", "Blue")},
    ("Paper_2", "Blue"): {"dur": 20, "prec": ("Paper_2", "Green")},
    ("Paper_2", "Green"): {"dur": 10, "prec": None},
    ("Paper_2", "Yellow"): {"dur": 34, "prec": ("Paper_2", "Blue")},
    ("Paper_3", "Blue"): {"dur": 12, "prec": ("Paper_3", "Yellow")},
    ("Paper_3", "Green"): {"dur": 17, "prec": ("Paper_3", "Blue")},
    ("Paper_3", "Yellow"): {"dur": 28, "prec": None},
}

TASKS = {
    ("T01", "HARD"): {"duration": 10, "successors": [("T01", "EASY")]},
    ("T02", "MEDIUM"): {"duration": 20, "successors": [("T02", "MEDIUM")]},
}


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
        # filter=lambda model, j, m, k, n: (k,n) == tasks[(j,m)]['prec']
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
    model.start = Var(model.TASKS, bounds=(0, ub))

    # Create ojective
    model.objective = Objective(expr=model.makespan, sense=minimize)

    # Constraint that tasks must finish before end
    model.finish = Constraint(
        model.TASKS,
        rule=lambda model, j, m: model.start[j, m] + model.dur[j, m] <= model.makespan,
    )

    #
    model.preceding = Constraint(
        model.TASK_ORDER,
        rule=lambda model, j, m, k, n: model.start[j, m] + model.dur[j, m]
        <= model.start[k, n],
    )

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
