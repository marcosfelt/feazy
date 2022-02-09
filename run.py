from datetime import datetime, timedelta, time, date
from feazy.plan import (
    Task,
    TaskGraph,
    read_file,
    optimize_schedule,
    download_notion_tasks,
    update_notion_tasks,
)
from gcsa.google_calendar import GoogleCalendar
import pytz
from beautiful_date import *
from google.auth.exceptions import RefreshError
import os
from pathlib import Path

import logging


def test_optimization(main_calendar="kobi.c.f@gmail.com"):
    # Read in tasks and dependencies
    task_list = read_file("data/tasks.txt")
    dependencies = read_file("data/dependencies.txt")

    # Put tasks in object
    tasks = TaskGraph(
        [
            Task(
                task_id=task[0],
                description=task[1],
                duration=timedelta(hours=int(task[2])),
            )
            for task in task_list
        ]
    )

    # Add depdencies
    for dependency in dependencies:
        tasks.add_dependency(dependency[0], dependency[1])

    # Start time and deadline
    timezone = pytz.timezone("Europe/London")
    start_time = timezone.localize((May / 1 / 2022)[00:00])
    deadline = start_time + 21 * days

    # Work Times (9-5 M-F)
    work_times = {
        i: [time(hour=9, minute=0, second=0), time(hour=17, minute=0, second=0)]
        for i in range(6)
    }

    # Google calendar for availabilities
    try:
        calendar = GoogleCalendar()
    except RefreshError:
        p = Path("~/.credentials/token.pickle")
        os.remove(p)
        calendar = GoogleCalendar()
    new_tasks = optimize_schedule(
        tasks,
        work_times=work_times,
        start_time=start_time,
        deadline=deadline,
        base_calendar=calendar,
        exclude_calendar_ids=[
            # Weather calendar
            "jc1o00r4ve65t348l20l2q0090ken3q7@import.calendar.google.com",
            # chandler's calendar
            "jcgsville@gmail.com",
            # Birthdays
            "addressbook#contacts@group.v.calendar.google.com",
        ],
        block_duration=timedelta(hours=2),
    )
    print(new_tasks)


def notion_task_optimization(base_calendar="kobi.c.f@gmail.com"):
    logger = logging.getLogger(__name__)

    # Start time and deadline
    timezone = pytz.timezone("Europe/London")
    start_time = timezone.localize((Feb / 7 / 2022)[00:00])
    deadline = timezone.localize((Jan / 31 / 2023)[00:00])

    # Read in tasks and dependencies
    logging.debug("Downloading tasks from Notion")
    database_id = "89357b5cf7c749d6872a32636375b064"
    tasks = download_notion_tasks(database_id, start_time)

    # Work Times (9-5 M-F)
    work_times = {
        i: [time(hour=9, minute=0, second=0), time(hour=14, minute=0, second=0)]
        for i in range(6)
    }

    # Google calendar for availabilities
    try:
        calendar = GoogleCalendar(base_calendar)
    except RefreshError:
        p = Path.home() / ".credentials" / "token.pickle"
        os.remove(p)
        calendar = GoogleCalendar()

    # Optimize scheudle
    new_tasks = optimize_schedule(
        tasks,
        work_times=work_times,
        start_time=start_time,
        deadline=deadline,
        base_calendar=calendar,
        exclude_calendar_ids=[
            # Weather calendar
            "jc1o00r4ve65t348l20l2q0090ken3q7@import.calendar.google.com",
            # chandler's calendar
            "jcgsville@gmail.com",
            # Birthdays
            "addressbook#contacts@group.v.calendar.google.com",
        ],
        block_duration=timedelta(hours=1),
    )

    # Update schedule in notion
    print(new_tasks)
    # update_notion_tasks(new_tasks, use_async=True)


if __name__ == "__main__":
    # timezone = pytz.timezone("Europe/London")
    # start_time = timezone.localize((Feb / 7 / 2022)[00:00])
    # database_id = "89357b5cf7c749d6872a32636375b064"
    # tasks = download_notion_tasks(database_id=database_id, start_time=start_time)
    # t = tasks.all_tasks[0]
    # t.scheduled_start = date(2022, 2, 14)
    # t.scheduled_deadline = date(2022, 2, 28)
    # print(t)

    start = datetime.now()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    notion_task_optimization()
    end = datetime.now()
    delta = (end - start).total_seconds() / 60
    logging.info(f"Took {delta:.01f} minutes to run")
