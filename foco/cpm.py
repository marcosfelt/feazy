from .utils import Graph, GraphDirection
from queue import LifoQueue, SimpleQueue
from typing import List, Optional


class Task:
    def __init__(self, task_code: str, duration: int, description: Optional[str] = None)-> None:
        self._task_code = task_code
        self.duration = duration
        self.description = description
        self._early_start: int = 0
        self._late_start: int = None

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

class TaskGroup:
    def __init__(self, tasks: List[Task]):
        self._tasks = {
            task.task_code: task
            for task in tasks
        }
    
    def remove_task(self, task_code):
        if task_code in self._tasks:
            return self._tasks.pop(task_code)

    def __getitem__(self, task_code):
        return self._tasks.get(task_code)

    def __repr__(self) -> str:
        # headers = ["Task Code", "Duration", "Early Start", "Early Finish", "Late Start", "Late Finish", "Slack"]
        headers = ["Task Code", "Task Description"+" "*20]
        repr = "".join([f"{h}\t" for h in headers])+"\n"
        d = 2
        for task in self._tasks.values():
            values = [
                task.task_code, 
                task.description,
                # task.duration,
                # task.early_start, 
                # task.early_finish,
                # task._late_start,
                # task.late_finish, 
                # task.slack
            ]
            repr += "".join([str(v).ljust(len(h), " ")+"\t" for h, v in zip(headers,values)])
            offset = int(task.early_start/d if task.early_start != 0 else 0)
            repr += " "*offset
            # repr += "|"
            dur = int(task.duration/d if task.duration>1 else 1)
            repr += "|"*dur
            if task.late_start > task.early_finish:
                offset = int((task.late_start-task.early_finish)/d)
                repr += " "*offset
                repr += "*"*dur
            else:
                dur = int((task.late_finish-task.early_finish)/d)
                repr += "*"*dur
            repr += "\n"
        return repr
        
    @property
    def all_tasks(self):
        return [t for t in self._tasks]

def cpm(
    tasks: TaskGroup,
    dependencies: Graph,
    deadline: int,
    root_task_code: str,
    finish_task_code: Optional[str]="Finish"
):
    """Critical Path Method
    
    Arguments
    ---------
    tasks : TaskGroup
        The set of tasks to perform the critical path method on
    dependencies : Graph
        The dependencies between tasks
    deadline : int
        The deadline for the project to be complete
    root_task_code : str
        The code of the root task 
    finish_task_code : optional, str
        The code of the final task. Defaults to Finish.
    
    """
    if dependencies.edge_direction != GraphDirection.DIRECTED:
        raise ValueError("Dependencies must be a directed graph")

    # Breadth-first search for forward pass
    visit_queue = SimpleQueue() # FIFO queue for doing forward pass
    backward_visit_stack = LifoQueue() # Stack for doing backward wpass
    root_node = dependencies.get_node(root_task_code)
    predecessor_nodes = [root_node]
    for adj in root_node.adjacents:
        visit_queue.put(adj)
    early_finish_time = 0
    latest_predecessor = 0
    while not visit_queue.empty():
        current_node = visit_queue.get(block=True)
        # Find latest finish time of predecessors
        if current_node not in predecessor_nodes and current_node.val != finish_task_code:
            backward_visit_stack.put(current_node)
            
            # Find predecessors and calculate latest
            for predecessor_node in predecessor_nodes:
                if current_node in predecessor_node.adjacents:
                    adj_finish_time = tasks[predecessor_node.val].early_finish
                    if adj_finish_time > latest_predecessor:
                        latest_predecessor = adj_finish_time

            # Make this task's early start time 
            # the latest finish of its predecessors
            task = tasks[current_node.val]
            task.early_start = latest_predecessor

            # Add new adjacents
            for adj in current_node.adjacents:
                if adj not in predecessor_nodes:
                    visit_queue.put(adj, block=True)
            
            # Mark current node as visited
            predecessor_nodes.append(current_node)
       
            # Make the overall finish time the latest overall finishing task
            if task.early_finish > early_finish_time:
                early_finish_time = task.early_finish

    # Make Finish task last
    finish_task = tasks[finish_task_code]
    finish_task.early_start = early_finish_time
    finish_task.late_start = deadline

    # Breadth-first search for backward pass
    successor_nodes = [dependencies.get_node(finish_task_code)]
    first_dependent_time = deadline
    root_task = tasks[root_task_code]
    check_at_end = []
    while not backward_visit_stack.empty():
        # Make each upstream tasks’ late finish time 
        # the earliest late start of its successors
        current_node = backward_visit_stack.get(block=True)
        earliest_successor = deadline
        visited_successors = all([adj in successor_nodes for adj in current_node.adjacents])
        if visited_successors:
            for adj in current_node.adjacents:
                adj_start_time = tasks[adj.val].late_start
                if adj_start_time < earliest_successor:
                    earliest_successor = adj_start_time
            task = tasks[current_node.val]
            task.late_start = earliest_successor - task.duration
        
            # Make the root task the earliest late starter
            if task.late_start < first_dependent_time and task != root_task:
                first_dependent_time = task.late_start

            # Append current node to visted nodes
            successor_nodes.append(current_node)
        else:
            check_at_end.append(current_node)
        
        # Some nodes might need to be saved to the end
        if backward_visit_stack.empty():
            for node in check_at_end:
                backward_visit_stack.put(node)
            check_at_end = []

    # Make the root task the earliest late starter
    root_task.late_start = first_dependent_time - root_task.duration
