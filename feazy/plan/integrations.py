import pdb
from .task import Task, TaskGraph
from notion_client import Client, AsyncClient
from dotenv import load_dotenv
from datetime import timedelta, datetime
import pytz
import warnings
import asyncio
import os
from typing import Optional


load_dotenv()  # take environment variables from .env.
import logging


def connect_notion(logger, use_async=False) -> Client:
    # Connect notion
    token = os.environ.get("NOTION_TOKEN")
    if token is None:
        raise ValueError("No notion token found in environment varaibles")
    if use_async:
        return AsyncClient(auth=token, logger=logger)
    else:
        return Client(auth=token, logger=logger)


def download_notion_tasks(
    database_id: str,
    start_time: datetime,
    timezone="Europe/London",
) -> TaskGraph:
    logger = logging.getLogger(__name__)

    # Connect to notion
    notion = connect_notion(logger)

    # Get all uncompleted task
    logging.info("Querying notion for tasks")
    query = {"filter": {"property": "Complete", "checkbox": {"equals": False}}}
    results = notion.databases.query(database_id=database_id, **query).get("results")
    if results is None:
        raise ValueError("No results from Notion")

    # from pprint import pprint

    # pprint(results[0])
    # import pdb

    # pdb.set_trace()

    # Create tasks
    tasks = TaskGraph()
    date_fmt = "%Y-%m-%d"
    tz = pytz.timezone(timezone)
    for result in results:
        extracted_props = {}
        extracted_props["task_id"] = result["id"]
        props = result["properties"]

        # Assert that it is not complete
        is_complete = props["Complete"]["checkbox"]
        assert not is_complete

        # Description
        to_do = props.get("To-Do")
        if to_do:
            extracted_props["description"] = to_do["title"][0]["plain_text"]

        # Duration
        duration = props.get("Duration (hours)")
        if duration:
            dur = duration.get("number")
            dur = dur if dur else 0
            extracted_props["duration"] = timedelta(hours=dur)

        # Earliest Start
        extracted_props["earliest_start"] = extract_date_property(
            props, "Earliest Start", start_time, tz=tz
        )
        # if extracted_props["earliest_start"] is None:
        #     from pprint import pprint

        #     pprint(props)
        #     import pdb

        #     pdb.set_trace()

        # Deadline
        extracted_props["deadline"] = extract_date_property(
            props, "Deadline", start_time, tz=tz
        )

        # Wait time
        wait_time = props.get("Wait Time (days)")
        if wait_time:
            w = wait_time.get("number")
            w = w if w else 0
            extracted_props["wait_time"] = timedelta(days=w)

        # Scheduled events
        # extracted_props["scheduled_early_start"] = extract_date_property(
        #     props, "Scheduled Early Start", start_time, tz=tz
        # )
        # extracted_props["scheduled_late_start"] = extract_date_property(
        #     props, "Scheduled Late Start", start_time, tz=tz
        # )
        # extracted_props["scheduled_early_finish"] = extract_date_property(
        #     props, "Scheduled Early Finish", start_time, tz=tz
        # )
        # extracted_props["scheduled_deadline"] = extract_date_property(
        #     props, "Scheduled Due Date", start_time, tz=tz
        # )

        # Create task
        if duration and to_do:
            tasks.add_task(Task(**extracted_props))
        else:
            warnings.warn(
                f"""Did not add {extracted_props["task_id"]} because missing description or duration."""
            )

    # Add task depdendencies
    for result in results:
        task_id = result["id"]
        relations = result["properties"]["Predecessors"]["relation"]
        if len(relations) > 0:
            predecessors = [r["id"] for r in relations]
            for predecessor in predecessors:
                if predecessor in tasks._nodes:
                    tasks.add_dependency(predecessor, task_id)
    logging.info(f"Retrieved {len(tasks._nodes)} from Notion")
    return tasks


def extract_date_property(
    props: dict,
    property: str,
    start_time: datetime,
    tz=None,
    date_format_str: Optional[str] = None,
):
    d = props.get(property)
    d = d["date"] if d is not None else d
    if tz is not None and d is not None:
        date_format_str = "%Y-%m-%d" if date_format_str is None else date_format_str
        d = tz.localize(datetime.strptime(d["start"], date_format_str))
    if d:
        if d < start_time:
            raise ValueError(f"""Deadline for task before start time for project.""")

    return d


fmt_date = lambda d: d.strftime("%Y-%m-%d")


async def _update_notion_tasks(tasks: TaskGraph):
    """"""
    logger = logging.getLogger(__name__)

    # Connect to notion
    async_notion = connect_notion(logger, use_async=True)

    # Update notion tasks
    requests = []
    for task in tasks.all_tasks:
        if task.scheduled_early_start or task.scheduled_deadline:
            props = {
                "Scheduled Early Start": {
                    "date": {"start": fmt_date(task.scheduled_early_start)}
                    if task.scheduled_early_start
                    else None
                },
                "Scheduled Late Start": {
                    "date": None,
                },
                "Scheduled Early Finish": {
                    "date": {"start": fmt_date(task.scheduled_early_finish)}
                    if task.scheduled_early_finish
                    else None
                },
                "Scheduled Due Date": {
                    "date": {"start": fmt_date(task.scheduled_deadline)}
                    if task.scheduled_deadline
                    else None
                },
            }
            requests.append(
                async_notion.pages.update(
                    **{"page_id": task.task_id, "properties": props}
                )
            )

    # Send requests
    logging.info(f"Updating {len(requests)} tasks in Notion")
    await asyncio.gather(*requests)


def update_notion_tasks(tasks: TaskGraph, use_async=False):
    if use_async:
        asyncio.run(_update_notion_tasks(tasks))
    else:
        logger = logging.getLogger(__name__)

        # Connect to notion
        notion = connect_notion(logger, use_async=False)
        logging.info(f"Updating {len(tasks._nodes)} tasks in Notion")
        for task in tasks.all_tasks:
            props = {
                "Scheduled Early Start": {
                    "date": {"start": fmt_date(task.scheduled_early_start)}
                },
                "Scheduled Late Start": {
                    "date": {"start": fmt_date(task.scheduled_late_start)}
                },
                "Scheduled Early Finish": {
                    "date": {"start": fmt_date(task.scheduled_early_finish)}
                },
                "Scheduled Due Date": {
                    "date": {"start": fmt_date(task.scheduled_deadline)}
                },
            }
            notion.pages.update(**{"page_id": task.task_id, "properties": props})


def get_page_url(page_id: str):
    # Connect notion
    token = os.environ.get("NOTION_TOKEN")
    if token is None:
        raise ValueError("No notion token found")
    notion = Client(auth=token)

    page = notion.pages.retrieve(page_id)

    return page["url"]
