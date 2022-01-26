from __future__ import annotations
from uuid import uuid4
from .utils import Node, Graph, GraphDirection
from typing import List, Optional, Tuple, Union
from datetime import date, datetime, time
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
    difficulty : `TaskDifficulty`
        Difficulty level of the task
    duration : int
        Taks duration in minutes.
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

    """

    def __init__(
        self,
        difficulty: TaskDifficulty,
        duration: int,
        description: Optional[str],
        task_id: Optional[str] = None,
        earliest_start: Union[date, datetime] = None,
        deadline: Union[date, datetime] = None,
    ) -> None:
        if task_id is None:
            task_id = uuid4()
        super().__init__(val=task_id)
        self._difficulty = difficulty
        self.duration = duration
        self.description = description
        self._early_start: int = 0
        self._late_start: int = 0
        self._earliest_start = earliest_start
        self._deadline = deadline
        self._scheduled_start: datetime = None
        self._scheduled_finish: datetime = None

    @property
    def task_id(self):
        return self.val

    @property
    def early_start(self) -> int:
        """The early start in task days"""
        return self._early_start

    @early_start.setter
    def early_start(self, time: int) -> None:
        self._early_start = time

    @property
    def early_finish(self) -> int:
        return self._early_start + self.duration

    @property
    def late_start(self) -> int:
        return self._late_start

    @late_start.setter
    def late_start(self, time: int) -> None:
        self._late_start = time

    @property
    def late_finish(self) -> int:
        return self._late_start + self.duration

    @property
    def earliest_start(self) -> datetime:
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
    def difficulty(self):
        return self._difficulty

    # def convert_deadline_to_task(self, converter: TaskTimeConverter):
    #     """Convert deadline to actual task time"""
    #     # Make sure to cache
    #     pass

    @property
    def slack(self) -> int:
        return self.late_start - self.early_start

    @property
    def successors(self) -> List[Task]:
        return self.adjacents


class TaskGraph(Graph):
    """Representation of a graphs of tasks"""

    def __init__(self, tasks: List[Task]):
        super().__init__(edge_direction=GraphDirection.DIRECTED)
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

    def add_dependency(
        self, source_task_id: str, successor_task_id: str
    ) -> Tuple[Node, Node]:
        source_node = self.get_node(source_task_id)
        destination_node = self.get_node(successor_task_id)

        source_node.add_adjacent(destination_node)

        if self.edge_direction == GraphDirection.UNDIRECTED:
            destination_node.add_adjacent(source_node)

        return source_node, destination_node

    def __getitem__(self, task_id):
        return self._nodes.get(task_id)

    def __repr__(self) -> str:
        # headers = ["Task Code", "Duration", "Early Start", "Early Finish", "Late Start", "Late Finish", "Slack"]
        headers = ["Task Code", "Task Description" + " " * 20, "Difficulty", "Start"]
        repr = "".join([f"{h}\t" for h in headers]) + "\n"
        d = 2
        for task in self._nodes.values():
            values = [
                task.task_id,
                task.description,
                task.difficulty.name,
                # task.duration,
                task.early_start,
                # task.early_finish,
                # task._late_start,
                # task.late_finish,
                # task.slack
            ]
            repr += "".join(
                [str(v).ljust(len(h), " ") + "\t" for h, v in zip(headers, values)]
            )
            offset = int(task.early_start / d if task.early_start != 0 else 0)
            repr += " " * offset
            # repr += "|"
            dur = int(task.duration / d if task.duration > 1 else 1)
            repr += "|" * dur
            if task.late_start > task.early_finish:
                offset = int((task.late_start - task.early_finish) / d)
                repr += " " * offset
                repr += "*" * dur
            else:
                dur = int((task.late_finish - task.early_finish) / d)
                repr += "*" * dur
            repr += "\n"
        return repr

    @property
    def all_tasks(self, task_difficulty: Optional[TaskDifficulty] = None) -> List[Task]:
        if not task_difficulty:
            return [t for t in self._nodes.values()]
        else:
            return [t for t in self._nodes.values() if t.difficulty == task_difficulty]