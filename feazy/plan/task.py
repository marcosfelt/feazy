from __future__ import annotations
from uuid import uuid4
from .utils import Node, Graph, GraphDirection
from typing import List, Optional, Tuple, Union
from datetime import date, datetime, time, timedelta
import warnings
from enum import Enum


class TaskDifficulty(Enum):
    HARD = "Hard"
    MEDIUM = "Medium"
    EASY = "Easy"


class Task(Node):
    """Representation of a task

    Arguments
    ---------
    duration : timedelta
        Taks duration
    description : str
        Description of the task
    task_id : str, optional
        A unique identifier for the task. By default assigns a UUID4.
    earliest_start : date or datetime, optional
        The earliest time at which this task could start. Used in by scheduling algorithms.
        If not specified, defaults to tomorrow.
    deadline : date or datetime, optional
        The deadline for the task. If not specified, no deadline set and
        scheduling algorithm free to set finish of task.
    wait_time : timedelta, optional
        The amount of time to wait before scheduling an successor task. Useful
        for tasks that are dependent on some external input.s

    """

    def __init__(
        self,
        duration: timedelta,
        description: str,
        task_id: Optional[str] = None,
        earliest_start: Optional[Union[date, datetime]] = None,
        deadline: Union[date, datetime] = None,
        wait_time: Optional[timedelta] = None,
    ) -> None:
        if task_id is None:
            task_id = uuid4()
        super().__init__(val=task_id)
        self.duration = duration
        self.description = description
        # self._early_start: int = 0
        # self._late_start: int = 0
        self._earliest_start = earliest_start
        self._deadline = deadline
        self._wait_time = wait_time
        self._scheduled_start: datetime = None
        self._scheduled_deadline: datetime = None

    @property
    def task_id(self):
        return self.val

    # @property
    # def early_start(self) -> int:
    #     """The early start in task days"""
    #     return self._early_start

    # @early_start.setter
    # def early_start(self, time: int) -> None:
    #     self._early_start = time

    # @property
    # def early_finish(self) -> int:
    #     return self._early_start + self.duration

    # @property
    # def late_start(self) -> int:
    #     return self._late_start

    # @late_start.setter
    # def late_start(self, time: int) -> None:
    #     self._late_start = time

    # @property
    # def late_finish(self) -> int:
    #     return self._late_start + self.duration

    @property
    def earliest_start(self) -> Union[datetime, None]:
        if type(self._earliest_start) == date:
            return datetime.combine(
                self._earliest_start, time(hour=0, minute=0, second=0)
            )
        else:
            return self._earliest_start

    @property
    def deadline(self) -> date:
        if type(self._deadline) == date:
            return datetime.combine(self._deadline, time(hour=0, minute=0, second=0))
        else:
            return self._deadline

    @deadline.setter
    def deadline(self, day: date):
        if day < date.today():
            warnings.warn(f"{day} is before the today.")
        self._deadline = day

    @property
    def scheduled_start(self) -> Union[date, datetime, None]:
        return self._scheduled_start

    @scheduled_start.setter
    def scheduled_start(self, val):
        if type(val) in [date, datetime]:
            self._scheduled_start = val
        else:
            raise ValueError("Start time must be a date or datetime")

    @property
    def scheduled_dedadline(self) -> Union[date, datetime, None]:
        return self._scheduled_deadline

    @scheduled_dedadline.setter
    def scheduled_deadline(self, val):
        if type(val) in [date, datetime]:
            self._scheduled_deadline = val
        else:
            raise ValueError("Scheduled deadline must be a date or datetime")

    @property
    def wait_time(self) -> timedelta:
        if self._wait_time is not None:
            return self._wait_time
        else:
            return timedelta()

    @property
    def successors(self) -> List[Task]:
        return self.adjacents


class TaskGraph(Graph):
    """Representation of a graphs of tasks"""

    def __init__(self, tasks: Optional[List[Task]] = None):
        super().__init__(edge_direction=GraphDirection.DIRECTED)
        if tasks is not None:
            for task in tasks:
                self._nodes[task.val] = task

    def add_task(self, task: Task):
        if task.val in self._nodes:
            raise KeyError(f"Task with f{task.val} already exists in graph")
        else:
            self._nodes[task.val] = task

    def remove_task(self, task_id):
        if task_id in self._tasks:
            return self._tasks.pop(task_id)

    def remove_dependency(self, source_task_id: str, successor_task_id: str):
        self.remove_edge(source_task_id, successor_task_id)

    def add_dependency(
        self, source_task_id: str, successor_task_id: str
    ) -> Tuple[Node, Node]:
        if source_task_id == successor_task_id:
            raise ValueError("A task cannot be a predecessor of itself.")
        source_node = self.get_node(source_task_id)
        destination_node = self.get_node(successor_task_id)

        source_node.add_adjacent(destination_node)

        if self.edge_direction == GraphDirection.UNDIRECTED:
            destination_node.add_adjacent(source_node)

        return source_node, destination_node

    def __getitem__(self, task_id) -> Task:
        return self._nodes.get(task_id)

    def __repr__(self) -> str:
        headers = ["Task ID", "Task Description" + " " * 20, "Start"]
        repr = "".join([f"{h}\t" for h in headers]) + "\n"

        scheduled_tasks = [
            t for t in self.all_tasks if t.scheduled_start and t.scheduled_deadline
        ]
        scheduled_tasks.sort(key=lambda t: t.scheduled_deadline, reverse=False)
        end_time: datetime = scheduled_tasks[-1].scheduled_deadline
        scheduled_tasks.sort(key=lambda t: t.scheduled_start, reverse=False)
        start_time: datetime = scheduled_tasks[0].scheduled_start
        d = int((end_time - start_time).total_seconds() / (3600 * 24) / 50)
        d = 1 if d == 0 else d
        for task in scheduled_tasks:
            values = [task.task_id, task.description, task.scheduled_start]
            repr += "".join(
                [str(v).ljust(len(h), " ") + "\t" for h, v in zip(headers, values)]
            )
            offset = int(
                (task.scheduled_start - start_time).total_seconds() / (3600 * 24) / d
            )  # in days
            repr += " " * offset
            repr += "|"
            dur = int(task.duration.total_seconds() / (3600 * 24) / d)  # In days
            repr += "|" * dur
            dur
            repr += "\n"
        return repr

    @property
    def all_tasks(self, task_difficulty: Optional[TaskDifficulty] = None) -> List[Task]:
        if not task_difficulty:
            return [t for t in self._nodes.values()]
        else:
            return [t for t in self._nodes.values() if t.difficulty == task_difficulty]
