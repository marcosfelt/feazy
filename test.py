from foco.utils import Graph, GraphDirection, read_file
from foco.cpm import Task, TaskGroup, cpm

def main():
    # Read in tasks and dependencies
    task_list = read_file("data/tasks.txt")
    dependencies=read_file("data/dependencies.txt")

    # Put tasks in object
    tasks = TaskGroup([
        Task(
            task_code=task[0],
            description=task[1],
            duration=int(task[2])
        )
        for task in task_list
    ])

    # Create depdency graph
    g = Graph(edge_direction=GraphDirection.DIRECTED)
    for dependency in dependencies:
        g.add_edge(dependency[0], dependency[1])

    # Run critical path method
    cpm(
        tasks,
        g,
        deadline=100,
        root_task_code="T03"
    )

    # Print out final task group
    print(tasks)

if __name__ == "__main__":
    main()