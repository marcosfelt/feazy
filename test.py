from datetime import datetime, timedelta, time
from feazy.plan import Task, TaskGraph, read_file, optimize_schedule
from gcsa.google_calendar import GoogleCalendar
import pytz
from beautiful_date import *
from google.auth.exceptions import RefreshError
import os
from pathlib import Path

import logging


def main_optimization(main_calendar="kobi.c.f@gmail.com"):
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
    timezone = pytz.timezone("UTC")
    start_time = timezone.localize((May / 1 / 2022)[00:00])
    deadline = start_time + 14 * days

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


if __name__ == "__main__":
    start = datetime.now()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main_optimization()
    end = datetime.now()
    delta = (end - start).total_seconds() / 60
    logging.info(f"Took {delta:.01f} minutes to run")

    [
        "gm4r1jfef9c5mhncikq3f21tquoq7mkl@import.calendar.google.com",  # deleted
        "kobi.c.f@gmail.com",  # yes
        "qj19opnssbjsdn1ss8nlmmrkcgo4cim2@import.calendar.google.com",  # yes
        "addressbook#contacts@group.v.calendar.google.com",  # no
        "1sgehna3vnjore0l378nfvnkk4@group.calendar.google.com",  # yes
        "gp4vjf4ln68kt692bv3rllgq8c@group.calendar.google.com",  # yes
        "kcmf2@cam.ac.uk",  # yes
    ]
