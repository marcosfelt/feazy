from __future__ import print_function
from copy import deepcopy
from gcsa.google_calendar import GoogleCalendar, Event
from gcsa.event import Transparency
from beautiful_date import *
from .task import Task, TaskGraph
from notion_client import Client, AsyncClient
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from typing import List
import pickle
from dotenv import load_dotenv
from datetime import timedelta, datetime, date, time
from pyrfc3339 import parse as rfc_parse
import pytz
import warnings
import asyncio
import os
from typing import Optional, List, Tuple, Dict, Union
import logging

load_dotenv()  # take eimport logging

DEFAULT_TIMEZONE = "Europe/London"


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
    transparent_as_free: bool = True,
    split_across_days: bool = True,
    default_timezone: Optional[str] = DEFAULT_TIMEZONE,
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
    logger = logging.getLogger(__name__)
    # Get events in that time range
    events = []
    if type(calendars) == GoogleCalendar:
        calendars = [calendars]
    logger.debug("Downloading events")
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
    convert_dates_to_datetimes(events, default_timezone)
    events = _remove_duplicates(events)
    events.sort()

    # Specify availability as a list of times where there aren't events
    availabilities = []
    logger.debug("Calculating avaialabilities")
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
    if latest_end < end_time:
        availability = (
            _split_across_days((events[-1].end, end_time))
            if split_across_days
            else [(latest_end, end_time)]
        )
        availabilities.extend(availability)
    return availabilities


def _remove_duplicates(events: List[Event]):
    new_events = events[:1]
    for event in events:
        exists = False
        for other in new_events:
            if event.start == other.start and event.end == other.end:
                exists = True
        if not exists:
            new_events.append(event)

    return new_events


def _split_across_days(
    availability: Tuple[datetime, datetime]
) -> List[Tuple[datetime, datetime]]:
    """Split an availability window at 11:59:59 each day"""

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


def convert_dates_to_datetimes(events: List[Event], default_timezone: str):
    timezone = pytz.timezone(default_timezone)
    for event in events:
        if type(event.start) == date:
            event.start = timezone.localize(
                datetime.combine(
                    event.start,
                    time(
                        0,
                        0,
                    ),
                )
            )
        if type(event.end) == date:
            event.end = timezone.localize(
                datetime.combine(event.end - timedelta(days=1), time(23, 59, 59))
            )


def connect_notion(logger, use_async=False) -> Client:
    """Connect to Notion API"""
    token = os.environ.get("NOTION_TOKEN")
    if token is None:
        raise ValueError("No notion token found in environment varaibles")
    if use_async:
        return AsyncClient(auth=token, logger=logger)
    else:
        return Client(auth=token, logger=logger)


def download_notion_tasks(
    database_id: str,
    timezone="Europe/London",
) -> TaskGraph:
    """Download tasks from Notion

    Arguments
    --------
    database_id : str
        id of the database with your tasks
    start_time : datetime
        Start time for tasks
    timezone: str
        Timezone to put dates in. Defaults to "Europe/London"

    """
    logger = logging.getLogger(__name__)

    # Connect to notion
    notion = connect_notion(logger)

    # Get all uncompleted tasks
    logging.info("Querying notion for tasks")
    # query = {"filter": {"property": "Complete", "checkbox": {"equals": False}}}
    results = notion.databases.query(database_id=database_id, **{}).get("results")
    if results is None:
        raise ValueError("No results from Notion")

    # from pprint import pprint

    # pprint(results[0])
    # import pdb

    # pdb.set_trace()

    # Create tasks
    tasks = TaskGraph()
    tz = pytz.timezone(timezone)
    previous_ids = []
    project_names = {}
    for result in results:
        extracted_props = {}
        extracted_props["task_id"] = result["id"]
        assert not extracted_props["task_id"] in previous_ids
        previous_ids.append(extracted_props["task_id"])
        props = result["properties"]

        # Completion status
        extracted_props["completed"] = props["Complete"]["checkbox"]

        # Description
        to_do = props.get("To-Do")
        if to_do:
            extracted_props["description"] = to_do["title"][0]["plain_text"]

        # Project
        project_id = props["Project"]["relation"]
        if project_id:
            project_id = project_id[0]["id"]
            project_name = project_names.get(project_id)
            if project_name is None:
                project_name = get_project_name(project_id, notion)
                project_names[project_id] = project_name
            extracted_props["project"] = project_name

        # Duration
        duration = props.get("Duration (hours)")
        if duration:
            dur = duration.get("number")
            dur = dur if dur else 0
            extracted_props["duration"] = timedelta(hours=dur)

        # Earliest Start
        extracted_props["earliest_start"] = extract_date_property(
            props, "Earliest Start", tz=tz
        )

        # Deadline
        extracted_props["deadline"] = extract_date_property(props, "Deadline", tz=tz)

        # Wait time
        wait_time = props.get("Wait Time (days)")
        if wait_time:
            w = wait_time.get("number")
            w = w if w else 0
            extracted_props["wait_time"] = timedelta(days=w)

        # Scheduled events
        extracted_props["scheduled_early_start"] = extract_date_property(
            props, "Scheduled Early Start", tz=tz
        )
        extracted_props["scheduled_late_start"] = extract_date_property(
            props, "Scheduled Late Start", tz=tz
        )
        extracted_props["scheduled_early_finish"] = extract_date_property(
            props, "Scheduled Early Finish", tz=tz
        )
        extracted_props["scheduled_deadline"] = extract_date_property(
            props, "Scheduled Due Date", tz=tz
        )

        # Gtasks ID
        gtasks_id = props.get("GoogleTaskId")
        if gtasks_id:
            if len(gtasks_id["rich_text"]) > 0:
                extracted_props["gtasks_id"] = gtasks_id["rich_text"][0]["plain_text"]
            else:
                extracted_props["gtasks_id"] = None

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

    # Check for cyclic dependencies
    if tasks.is_cyclic():
        cycles = "".join(f"{cycle} | " for cycle in tasks._cycles)
        print(tasks)
        raise ValueError(f"The following tasks are in cycles: {cycles}")

    logging.info(f"Retrieved {len(tasks._nodes)} tasks from Notion")
    return tasks


def get_project_name(project_page_id: str, notion: Client):
    logger = logging.getLogger(__name__)
    logger.debug(f"Getting project page {project_page_id}")
    page = notion.pages.retrieve(project_page_id)
    title = page["properties"]["Name"]["title"]
    if len(title) > 0:
        name = title[0]["plain_text"]
    else:
        name = ""
    icon = page.get("icon")
    if icon:
        name = icon["emoji"] + "  " + name
    return name


def extract_date_property(
    props: dict,
    property: str,
    tz=None,
    date_format_str: Optional[str] = None,
):
    d = props.get(property)
    d = d["date"] if d is not None else d
    if tz is not None and d is not None:
        date_format_str = "%Y-%m-%d" if date_format_str is None else date_format_str
        d = tz.localize(datetime.strptime(d["start"], date_format_str))

    return d


fmt_date = lambda d: d.strftime("%Y-%m-%d")


async def _update_notion_tasks_async(request_bodies: List[Tuple[str, Dict]]):
    """Update tasks in Notion"""
    logger = logging.getLogger(__name__)

    # Connect to notion
    async_notion = connect_notion(logger, use_async=True)

    # Update notion tasks
    requests = [
        async_notion.pages.update(**{"page_id": task_id, "properties": props})
        for task_id, props in request_bodies
    ]
    # Send requests
    logging.info(f"Updating {len(requests)} tasks in Notion")
    await asyncio.gather(*requests)


def _update_notion_tasks(request_bodies: List[Tuple[str, Dict]]):
    """Update tasks in Notion"""
    logger = logging.getLogger(__name__)

    # Connect to notion
    notion = connect_notion(logger, use_async=False)

    # Update notion tasks
    logging.info(f"Updating {len(requests)} tasks in Notion")
    requests = [
        notion.pages.update(**{"page_id": task_id, "properties": props})
        for task_id, props in request_bodies
    ]


def update_notion_tasks(tasks: TaskGraph, use_async=False):
    request_bodies = []
    for task in tasks.all_tasks:
        if task.scheduled_early_start or task.scheduled_deadline:
            props = {
                "Scheduled Early Start": {
                    "date": {"start": fmt_date(task.scheduled_early_start)}
                    if task.scheduled_early_start
                    else None
                },
                "Scheduled Late Start": {
                    "date": {"start": fmt_date(task.scheduled_late_start)}
                    if task.scheduled_late_start
                    else None
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
                "GoogleTaskId": {
                    "rich_text": [
                        {"text": {"content": task.gtasks_id if task.gtasks_id else ""}}
                    ]
                },
            }
            request_bodies.append((task.task_id, props))

    if use_async:
        asyncio.run(_update_notion_tasks_async(request_bodies))
    else:
        _update_notion_tasks(request_bodies)


def get_notion_page_url(page_id: str):
    """Get a Notion page url from a page id"""
    # Connect notion
    token = os.environ.get("NOTION_TOKEN")
    if token is None:
        raise ValueError("No notion token found")
    notion = Client(auth=token)

    page = notion.pages.retrieve(page_id)

    return page["url"]


def _get_default_credentials_path() -> str:
    """Checks if ".credentials" folder in home directory exists. If not, creates it.
    :return: expanded path to .credentials folder
    """
    home_dir = os.path.expanduser("~")
    credential_dir = os.path.join(home_dir, ".credentials")
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, "credentials.json")
    return credential_path


def _get_credentials(
    token_path: str,
    credentials_dir: str,
    credentials_file: str,
    scopes: List[str],
    save_token: bool,
    host: str,
    port: int,
) -> Credentials:
    credentials = None

    if os.path.exists(token_path):
        with open(token_path, "rb") as token_file:
            credentials = pickle.load(token_file)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            credentials_path = os.path.join(credentials_dir, credentials_file)
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, scopes)
            credentials = flow.run_local_server(host=host, port=port)

        if save_token:
            with open(token_path, "wb") as token_file:
                pickle.dump(credentials, token_file)

    return credentials


def get_task_lists():
    """Shows basic usage of the Tasks API.
    Prints the title and ID of the first 10 task lists.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.

    SCOPES = ["https://www.googleapis.com/auth/tasks.readonly"]

    credentials_path = _get_default_credentials_path()
    credentials_dir, credentials_file = os.path.split(credentials_path)
    token_path = os.path.join(credentials_dir, "token.pickle")
    credentials = _get_credentials(
        token_path=token_path,
        credentials_dir=credentials_dir,
        credentials_file=credentials_file,
        scopes=SCOPES,
        save_token=True,
        host="localhost",
        port=8080,
    )
    try:
        service = build("tasks", "v1", credentials=credentials)

        # Call the Tasks API
        results = service.tasklists().list(maxResults=10).execute()
        items = results.get("items", [])

        if not items:
            print("No task lists found.")
            return

        print("Task lists:")
        for item in items:
            print("{0} ({1})".format(item["title"], item["id"]))
    except HttpError as err:
        print(err)


def get_gtasks_service() -> Resource:
    logger = logging.getLogger(__name__)

    SCOPES = ["https://www.googleapis.com/auth/tasks"]

    credentials_path = _get_default_credentials_path()
    credentials_dir, credentials_file = os.path.split(credentials_path)
    token_path = os.path.join(credentials_dir, "token.pickle")
    credentials = _get_credentials(
        token_path=token_path,
        credentials_dir=credentials_dir,
        credentials_file=credentials_file,
        scopes=SCOPES,
        save_token=True,
        host="localhost",
        port=8080,
    )
    try:
        service = build("tasks", "v1", credentials=credentials)
        return service
    except HttpError as err:
        logger.error(err)
        raise err


def get_gtasks(service, reclaim_list_id: str) -> List[Task]:
    logger = logging.getLogger(__name__)
    finished = False
    try:
        request = service.tasks().list(tasklist=reclaim_list_id)
        results = request.execute()
        items = results.get("items", [])
        while not finished:
            if results.get("nextPageToken"):
                request = service.tasks().list_next(request, results)
                results = request.execute()
                items.extend(results.get("items", []))
            else:
                finished = True
    except HttpError as err:
        logger.error(err)
        raise err

    return items


def sync_from_gtasks(tasks: TaskGraph, gtasks: List, copy=False) -> TaskGraph:
    """Update task completion status based on Gtasks"""
    if copy:
        tasks = deepcopy(tasks)

    # Set tasks completed in Gtasks but not in Notion to be complete in Notion
    # If the deadline has been delayed, update that
    for task in tasks.all_tasks:
        if task.gtasks_id:
            for gtask in gtasks:
                if gtask["id"] == task.gtasks_id:
                    # Completion status
                    completed = True if gtask["status"] == "completed" else False
                    if completed and not task.completed:
                        task.completed = True

                    # Due date update
                    due_date = rfc_parse(gtask["due"]) if gtask.get("due") else None
                    if due_date is None:
                        continue
                    elif due_date.date() >= task.scheduled_deadline:
                        task.scheduled_deadline = due_date.date()
    return tasks


def update_gtasks(
    tasks: TaskGraph, service: Resource, reclaim_list_id: str, copy=False
) -> TaskGraph:
    """Insert and update tasks in Gtasks

    Insert tasks with early starts less than 30 days from today
    Update tasks that have been changed

    Returns a tasks graph with update gtasks_ids

    """
    if copy:
        tasks = deepcopy(tasks)

    logger = logging.getLogger(__name__)
    batch = service.new_batch_http_request()
    today = datetime.today().date()

    def update_gtasks_id(request_id, response, exception):
        if exception is not None:
            logger.error(exception)
            raise exception
        else:
            gtasks_id = response["id"]
            task_id = response["notes"].lstrip("Notion ID: ")
            tasks[task_id].gtasks_id = gtasks_id

    for task in tasks.all_tasks:
        if not task.scheduled_early_start:
            continue
        dur = task.duration.total_seconds() / 3600
        due_date = fmt_date(task.scheduled_early_finish.date())
        start = fmt_date(task.scheduled_early_start.date())
        if not task.gtasks_id and (
            task.scheduled_early_start.date() - today
        ) <= timedelta(days=30):
            body = {
                "title": f"{task.description} ({task.project}) (type: work duration: {dur}h due: {due_date} notbefore: {start})",
                "notes": f"Notion ID: {task.task_id}",
            }
            batch.add(
                service.tasks().insert(tasklist=reclaim_list_id, body=body),
                callback=update_gtasks_id,
            )
        elif task.gtasks_id and task._changed:
            body = {
                "title": f"{task.description} ({task.project}) (type: work duration: {dur}h due: {due_date} notbefore: {start})",
            }
            batch.add(
                service.tasks().update(
                    tasklist=reclaim_list_id, task=task.gtasks_id, body=body
                )
            )
    batch.execute()
    return tasks
