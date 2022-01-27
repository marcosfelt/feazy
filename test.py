from tracemalloc import start
from feazy.plan import (
    Task,
    TaskGraph,
    TaskDifficulty,
    read_file,
    optimize_schedule,
    ScheduleBlockRule,
    get_calendars,
    get_availability,
)
from gcsa.google_calendar import GoogleCalendar
import pytz
from beautiful_date import *
import random

from feazy.plan.schedule import breakdown_tasks


def main_cpm():
    # Read in tasks and dependencies
    task_list = read_file("data/tasks.txt")
    dependencies = read_file("data/dependencies.txt")

    # Put tasks in object
    tasks = TaskGraph(
        [
            Task(task_code=task[0], description=task[1], duration=int(task[2]))
            for task in task_list
        ]
    )

    # Create depdency graph
    # g = Graph(edge_direction=GraphDirection.DIRECTED)
    # for dependency in dependencies:
    #     g.add_edge(dependency[0], dependency[1])

    # # Run critical path method
    # cpm(
    #     tasks,
    #     g,
    #     deadline=90,
    #     root_task_code="T03"
    # )

    # # Print out final task group
    # print(tasks)


def main_optimization(feazy_calendar_id: str):
    # Read in tasks and dependencies
    task_list = read_file("data/tasks.txt")
    dependencies = read_file("data/dependencies.txt")

    # Put tasks in object
    difficulties = [TaskDifficulty.HARD, TaskDifficulty.MEDIUM, TaskDifficulty.EASY]
    tasks = TaskGraph(
        [
            Task(
                task_id=task[0],
                description=task[1],
                duration=int(task[2]),
                difficulty=difficulties[random.randint(0, 2)],
            )
            for task in task_list
        ]
    )

    # Add depdencies
    for dependency in dependencies:
        tasks.add_dependency(dependency[0], dependency[1])

    # Optimize schedule using pyomo
    # optimize_schedule(tasks)

    # Get availability
    # base_calendar = GoogleCalendar("kobi.c.f@gmail.com")
    # all_calendars = get_calendars(
    #     base_calendar,
    #     only_selected=True,
    #     exclude=[
    #         "jc1o00r4ve65t348l20l2q0090ken3q7@import.calendar.google.com",
    #         feazy_calendar_id,
    #     ],
    # )
    timezone = pytz.timezone("UTC")
    start_time = timezone.localize((Jan / 24 / 2022)[00:00])
    end_time = start_time + 5 * days
    # availabilities = get_availability(
    #     all_calendars, start_time=start_time, end_time=end_time, split_across_days=True
    # )
    # print("Availabilities")
    # for availability in availabilities:
    #     fmt_date = lambda d: d.astimezone(pytz.timezone("UTC")).strftime(
    #         "%m/%d/%Y, %H:%M:%S"
    #     )
    #     print(fmt_date(availability[0]), "-", fmt_date(availability[1]))

    # # Schedule tasks
    dummy_start = (start_time - 1 * days).date()
    rules = {
        TaskDifficulty.HARD: ScheduleBlockRule(
            block_duration=45,
            break_duration_after=15,
            break_duration_before=15,
            max_blocks=2,
            dtstart=dummy_start,
        ),
        TaskDifficulty.MEDIUM: ScheduleBlockRule(
            block_duration=25,
            break_duration_after=5,
            max_blocks=8,
            dtstart=dummy_start,
        ),
        TaskDifficulty.EASY: ScheduleBlockRule(
            block_duration=25,
            break_duration_after=5,
            max_blocks=8,
            dtstart=dummy_start,
        ),
    }
    print(breakdown_tasks(tasks, rules))
    # feazy_calendar = GoogleCalendar(feazy_calendar_id)
    # schedule_tasks(
    #     feazy_calendar=feazy_calendar,
    #     availabilities=availabilities,
    #     tasks=tasks,
    #     schedule_rules=rules,
    # )


if __name__ == "__main__":
    main_optimization("n4j3l032kr821t4dvshof3rb74@group.calendar.google.com")

    # main_optimization()
