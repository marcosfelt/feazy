"""
1. Download all tasks from Notion

2. Merge in information from Reclaim
    - Sync reclaim completion status -> tasks downloaded from Notion
    - If due dates postponed in Reclaim/Google Tasks, update tasks downloaded from Notion

3. Optimization
    - Fix early and late starts for tasks already in Reclaim and try to solve
    - If that doesn't work, unfix and re-optimize. Note that Reclaim tasks were changed

4. Update Reclaim
    - If tasks already in Reclaim were changed, update them
    - Add tasks not in Reclaim with scheduled early starts within one month from now

5. Update everything Notion

"""

from datetime import datetime, timedelta, time
from feazy.plan import (
    optimize_schedule,
    download_notion_tasks,
    update_notion_tasks,
    get_gtasks,
    get_gtasks_service,
    sync_with_gtasks,
    update_gtasks,
)
import pytz
from beautiful_date import *
import logging
from typing import Optional

fmt_date = lambda d: d.astimezone(pytz.timezone("Europe/London")).strftime(
    "%m/%d/%Y, %H:%M:%S"
)


def task_optimization(
    notion_task_database_id: str,
    reclaim_tasklist_id: str,
    start_time: Optional[datetime] = None,
    deadline: Optional[datetime] = None,
    print_task_list=False,
):
    logger = logging.getLogger(__name__)

    # Download tasks from notion
    logger.info("Downloading tasks from Notion")
    tasks = download_notion_tasks(notion_task_database_id)

    # Start time and deadline
    timezone = pytz.timezone("Europe/London")
    if start_time is None:
        start_time = timezone.localize(datetime.today())
        for task in tasks.all_tasks:
            if (
                task.scheduled_early_start is not None
                and task.scheduled_deadline is not None
                and not task.completed
            ):
                if (
                    task.scheduled_early_start < start_time
                    and task.scheduled_deadline > start_time
                ):
                    start_time = task.scheduled_early_start
    if deadline is None:
        deadline = timezone.localize((Apr / 30 / 2023)[00:00])
    logger.info(f"Start Time: {fmt_date(start_time)}")
    logger.info(f"Deadline: {fmt_date(deadline)}")

    # Merge in Gtasks/Reclaim
    logger.info("Syncing Notion with Gtasks")
    gservice = get_gtasks_service()
    gtasks = get_gtasks(gservice, reclaim_tasklist_id)
    tasks = sync_with_gtasks(
        service=gservice,
        reclaim_list_id=reclaim_tasklist_id,
        gtasks=gtasks,
        tasks=tasks,
    )

    # Work Times (9-5 M-F)
    work_times = {
        i: [time(hour=9, minute=0, second=0), time(hour=14, minute=0, second=0)]
        for i in range(5)
    }
    work_times[5] = [
        time(hour=9, minute=0, second=0),
        time(hour=12, minute=0, second=0),
    ]

    # Remove completed tasks
    for task in tasks.all_tasks:
        if task.completed:
            tasks.remove_task(task.task_id)

    # Optimize schedule
    logger.info("Optimizing schedule")
    new_tasks = optimize_schedule(
        tasks,
        work_times=work_times,
        start_time=start_time,
        deadline=deadline,
        block_duration=timedelta(hours=1),
    )
    if print_task_list:
        print(new_tasks)

    # Update Gtasks/Reclaim
    logger.info("Updating Gtasks")
    new_tasks = update_gtasks(new_tasks, gservice, reclaim_tasklist_id)

    # Update schedule in notion
    update_notion_tasks(new_tasks, use_async=True)


if __name__ == "__main__":
    start = datetime.now()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    notion_task_database_id = "89357b5cf7c749d6872a32636375b064"
    reclaim_tasklist_id = "Nkw0V2wwWll6QVJ5a0hMUA"
    task_optimization(
        notion_task_database_id, reclaim_tasklist_id, print_task_list=True
    )
    end = datetime.now()
    delta = (end - start).total_seconds() / 60
    logging.info(f"Took {delta:.01f} minutes to run")
