from __future__ import annotations
from typing import List, Tuple, Dict
from queue import LifoQueue, SimpleQueue
import logging


class Node:
    """Representaiton of a node in a graph"""

    def __init__(self, val: str) -> None:
        self.val = val
        self._adjacents = []
        self._logger = logging.getLogger(__name__)

    def add_adjacent(self, node: Node) -> None:

        self._adjacents.append(node)

    def remove_adjacent(self, node: Node) -> None:
        try:
            self._adjacents.remove(node)
        except ValueError:
            raise ValueError("Node does not exist in adjacents")

    def is_adjacent(self, node: Node) -> bool:
        return node in self._adjacents

    @property
    def adjacents(self) -> List[Node]:
        self._logger.debug("Returned adjacent {")
        return self._adjacents

    def __repr__(self) -> str:
        adjs = "".join([f"{adj.val}, " for adj in self._adjacents]).rstrip(", ")
        return f"Node(Val: {self.val} | Adjacents: {adjs})"

    def __eq__(self, node: Node) -> bool:
        return (node.val == self.val) and all(
            [a1 == a2 for a1, a2 in zip(node.adjacents, self.adjacents)]
        )


class GraphDirection:
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


class GraphSearch:
    BFS = "bfs"
    DFS = "dfs"


class Graph:
    """Representation of a directed or undirected graph"""

    def __init__(self, edge_direction: str = GraphDirection.UNDIRECTED) -> None:
        self._nodes: Dict[str, Node]
        self._nodes = {}
        self.edge_direction = edge_direction
        self._cycles = []

    def create_node(self, val: str) -> Node:
        """Add a node to the graph"""
        if val in self._nodes:
            return self._nodes[val]
        else:
            node = Node(val)
            self._nodes[node.val] = node
            return node

    def remove_node(self, val: str) -> Node:
        """Remove a node from the graph"""
        if val in self._nodes:
            current = self._nodes.pop(val)
            for node in self._nodes.values():
                if current in node.adjacents:
                    node.remove_adjacent(current)
            return current

    def get_node(self, val: str) -> Node:
        return self._nodes.get(val)

    def add_edge(self, source_val: str, destination_val: str) -> Tuple[Node, Node]:
        """Add an edge to the graph"""
        source_node = self.create_node(source_val)
        destination_node = self.create_node(destination_val)

        source_node.add_adjacent(destination_node)

        if self.edge_direction == GraphDirection.UNDIRECTED:
            destination_node.add_adjacent(source_node)

        return source_node, destination_node

    def remove_edge(self, source_val: str, destination_val: str) -> Tuple[Node, Node]:
        source_node = self._nodes.get(source_val)
        destination_node = self._nodes.get(destination_val)
        if source_node and destination_node:
            source_node.remove_adjacent(destination_node)

            if self.edge_direction == GraphDirection.UNDIRECTED:
                destination_node.remove_adjacent(source_node)

        return source_node, destination_node

    def are_adjacent(self, source_val: str, destination_val: str) -> bool:
        """Check if two nodes are adjacent"""
        source_node = self._nodes.get(source_val)
        destination_node = self._nodes.get(destination_val)
        if source_node and destination_node:
            check_1 = destination_node in source_node.adjacents
            if self.edge_direction == GraphDirection.UNDIRECTED:
                check_2 = source_node in destination_node.adjacents
            return check_1 and check_2

    def graph_search(self, root_val: str, type: str = GraphSearch.BFS):
        """Search over the graph"""
        visited = {}
        visit_list = SimpleQueue() if type == GraphSearch.BFS else LifoQueue()

        # Add root to queue to start
        root = self._nodes.get(root_val)
        if root:
            visit_list.put(root)
        else:
            ValueError(f"{root_val} is not in the graph")

        while not visit_list.empty():
            node = visit_list.get(block=True)
            if node.val not in visited:
                yield node
                visited[node.val] = node
                for adjacent in node.adjacents:
                    visit_list.put(adjacent)

    @property
    def nodes(self) -> List[Node]:
        return [n for n in self._nodes.values()]

    def _is_cyclic(self, v, visited, recStack):

        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        recStack[v] = True

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        for neighbour in self._nodes[v].adjacents:
            if visited[neighbour.val] == False:
                if self._is_cyclic(neighbour.val, visited, recStack) == True:
                    self._cycles.append(neighbour.val)
                    return True
            elif recStack[neighbour.val] == True:
                return True

        # The node needs to be poped from
        # recursion stack before function ends
        recStack[v] = False
        return False

    # Returns true if graph is cyclic else false
    def is_cyclic(self):
        visited = {n: False for n in self._nodes}
        recStack = {n: False for n in self._nodes}
        for node in self._nodes:
            if visited[node] == False:
                if self._is_cyclic(node, visited, recStack) == True:
                    return True
        return False

    def __repr__(self) -> str:
        return (
            "Graph(\n" + "".join([f"\t{node}\n" for node in self._nodes.values()]) + ")"
        )


def reverse_graph(g: Graph):
    """Reverse direction of graph"""
    if g.edge_direction != GraphDirection.DIRECTED:
        raise ValueError("Must be a directed graph")
    new_graph = Graph(edge_direction=GraphDirection.DIRECTED)
    for node in g.nodes:
        for adj in node.adjacents:
            new_graph.add_edge(adj.val, node.val)
    return new_graph


def read_file(file: str, delimeter: str = "\t") -> List[List[str]]:
    with open(file, "r") as f:
        lines = f.readlines()

    return [line.rstrip("\n").split("\t") for line in lines]


if __name__ == "__main__":
    # Graphs
    g = Graph(GraphDirection.UNDIRECTED)
    g.create_node("a")
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("c", "d")
    g.add_edge("d", "f")

    print("Breadth First Search")
    for node in g.graph_search("a", type=GraphSearch.BFS):
        print(node)

    print()
    print("Depth First Search")
    for node in g.graph_search("a", type=GraphSearch.DFS):
        print(node)
