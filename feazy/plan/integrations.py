from multiprocessing.sharedctypes import Value
import pdb
from .task import Task, TaskGraph
import os
from notion_client import Client
from dotenv import load_dotenv
from datetime import timedelta, datetime
import pytz


load_dotenv()  # take environment variables from .env.
import logging


def download_notion_tasks(
    database_id: str,
    start_time: datetime,
    timezone="Europe/London",
):
    logger = logging.getLogger(__name__)

    # Connect notion
    token = os.environ.get("NOTION_TOKEN")
    if token is None:
        raise ValueError("No notion token found")
    notion = Client(auth=token, logger=logger)

    # Get all uncompleted tasks
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
        earliest_start = props["Earliest Start"]["date"]
        if earliest_start:
            d = tz.localize(datetime.strptime(earliest_start["start"], date_fmt))
            if d < start_time:
                d = start_time
            extracted_props["earliest_start"] = d
        else:
            extracted_props["earliest_start"] = None

        # Deadline
        deadline = props["Deadline"]["date"]
        if deadline:
            d = tz.localize(datetime.strptime(deadline["start"], date_fmt))
            if d < start_time:
                raise ValueError(
                    f"""Deadline before start time for task "{extracted_props['description']}" ({get_page_url(extracted_props['task_id'])})"""
                )
            extracted_props["deadline"] = d
        else:
            extracted_props["deadline"] = None

        # Create task
        tasks.add_task(Task(**extracted_props))

    # Add task depdendencies
    for result in results:
        task_id = result["id"]
        relations = result["properties"]["Predecessors"]["relation"]
        if len(relations) > 0:
            predecessors = [r["id"] for r in relations]
            for predecessor in predecessors:
                if predecessor in tasks._nodes:
                    tasks.add_dependency(predecessor, task_id)

    return tasks


def get_page_url(page_id: str):
    # Connect notion
    token = os.environ.get("NOTION_TOKEN")
    if token is None:
        raise ValueError("No notion token found")
    notion = Client(auth=token)

    page = notion.pages.retrieve(page_id)

    return page["url"]
