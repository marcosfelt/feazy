from utils import Graph, GraphDirection, GraphSearch
from typing import List, Optional, Dict


def read_file(file: str, delimeter: str="\t")-> List[List[str]]:
    with open(file, "r") as f:
        lines = f.readlines()

    return [line.rstrip("\n").split("\t") for line in lines]

class Task:
    def __init__(self, task_code: str, duration: int, description: Optional[str] = None)-> None:
        self._task_code = task_code
        self.duration = duration
        self.description = description
        self._early_start: int = 0
        self._late_start: int = 0

    @property
    def early_start(self)-> int:
        return self._early_start

    @early_start.setter
    def early_start(self, time: int) -> None:
        self._early_start = time

    @property
    def early_finish(self)-> int:
        return self._early_start + self.duration

    @property
    def late_start(self)-> int:
        return self._late_start

    @late_start.setter
    def late_start(self, time: int) -> None:
        self._late_start = time

    @property
    def late_finish(self)-> int:
        return self._late_start + self.duration

    @property
    def task_code(self)-> str:
        return self._task_code

    @property
    def slack(self) -> int:
        return self.late_start - self.early_start

# class TaskGroup:
#     def __init__(self, tasks: List[Task]):
#         self._tasks = {
#             task.task_code: task
#             for task in tasks
#         }

#     def get_task(self, task_code):
#         return self._tasks.get(task_code)
    
#     def remove_task(self, task_code):
#         if task_code in self._tasks:
#             return self._tasks.pop(task_code)

#     def __get

def cpm(
    deadline: int,
    root_task_code: Optional[str] = None, 
    finish_task_code: Optional[str]="Finish"
):
    """Critical Path Method"""
    # Read in tasks and dependencies
    task_list = read_file("tasks.txt")
    dependencies=read_file("dependencies.txt")

    # Put tasks in object
    tasks: Dict[str: Task]
    tasks = {
        task[0]: Task(
            task_code=task[0],
            description=task[1],
            duration=int(task[2])
        )
        for task in task_list
    }

    # Create depdency graph
    g = Graph(edge_direction=GraphDirection.UNDIRECTED)
    for dependency in dependencies:
        g.add_edge(dependency[0], dependency[1])
    
    # BFS for forward pass
    if root_task_code is None:
        root_task_code = task_list[0][0]
    root_task: Task
    root_task = tasks[root_task_code]
    previous_level_finish_time = root_task.early_finish
    this_level_finish_time = 0
    for node in g.graph_search(root_task_code, type=GraphSearch.BFS):
        if node.val == root_task_code:
            continue
        # Make each downstream tasks’ early start time the latest finish of its predecessors
        tasks[node.val].early_start = previous_level_finish_time

        # If finish time is greater than latest finish time of this level, make it the latest
        if tasks[node.val].early_finish > this_level_finish_time:
            this_level_finish_time = tasks[node.val].early_finish

        # When finished with a level, reset
        if node.val not in g._nodes[root_task_code].adjacents:
            root_task_code = node.val
            previous_level_finish_time = this_level_finish_time

    # Make sure Finish is always last
    finish_task = tasks[finish_task_code]
    finish_task.early_start = this_level_finish_time
    finish_task.late_start = deadline

    # # BFS for backward pass
    this_level_start_time = finish_task.late_start
    next_level_start_time = deadline
    for node in g.graph_search(finish_task_code, type=GraphSearch.BFS):
        if node.val == "Finish":
            continue

        # Calculate the late start time for each predecessor as T - t
        tasks[node.val].late_start = this_level_start_time - tasks[node.val].duration

        # Saave the earliest late start time
        if tasks[node.val].late_start < next_level_start_time:
            next_level_start_time = tasks[node.val].late_start

        # When finished with a level, reset
        if node.val not in g._nodes[root_task_code].adjacents:
            root_task_code = node.val
            this_level_start_time = next_level_start_time


    headers = ["Task Code", "Duration", "Early Start", "Early Finish", "Late Start", "Late Finish", "Slack"]
    print("".join([f"{h}\t" for h in headers]))
    for task in tasks.values():
        values = [task.task_code, task.duration, task.early_start, task.early_finish, task._late_start, task.late_finish, task.slack]
        line = "".join([str(v).ljust(len(h), " ")+"\t" for h, v in zip(headers,values)])
        line += "\t"
        offset = int(task.early_start/2 if task.early_start != 0 else 0)
        dur = int(task.duration/2 if task.duration !=0 else 1)
        line += " "*offset
        line += "|"*dur
        print(line)


if __name__ == "__main__":
    cpm(150, root_task_code="T03")