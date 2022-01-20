from foco.plan import Task, TaskGraph, TaskDifficulty, read_file, optimize_schedule
import random

from foco.plan.schedule import get_calendars


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


def main_optimization():
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
    optimize_schedule(tasks)

    # Print out final task group
    print(tasks)


if __name__ == "__main__":
    from foco.plan import get_availability
    from beautiful_date import *
    from gcsa.google_calendar import GoogleCalendar
    import pytz

    base = GoogleCalendar("kobi.c.f@gmail.com")
    all_calendars = get_calendars(
        base,
        only_selected=True,
        exclude=["jc1o00r4ve65t348l20l2q0090ken3q7@import.calendar.google.com"],
    )
    timezone = pytz.timezone("UTC")
    start_time = timezone.localize((Jan / 19 / 2022)[00:00])
    end_time = start_time + 1 * days
    availabilities = get_availability(
        all_calendars, start_time=start_time, end_time=end_time
    )
    for availability in availabilities:
        fmt_date = lambda d: d.astimezone(pytz.timezone("UTC")).strftime(
            "%m/%d/%Y, %H:%M:%S"
        )
        print(fmt_date(availability[0]), "-", fmt_date(availability[1]))

    # main_optimization()
