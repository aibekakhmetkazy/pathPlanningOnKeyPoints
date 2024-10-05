import math
import heapq


# Helper function to calculate the Euclidean distance between two nodes
def euclidean_distance(node1, node2, coordinates):
    x1, y1 = coordinates[node1]
    x2, y2 = coordinates[node2]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Dijkstra-based algorithm for pathfinding with a priority queue (can adapt for D* Lite)
def dijkstra(adjacency_list, start, goal, coordinates):
    # Priority queue: stores (distance, node)
    pq = [(0, start)]
    distances = {start: 0}
    previous_nodes = {start: None}
    visited = set()

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue
        visited.add(current_node)

        # Goal check
        if current_node == goal:
            break

        # Explore neighbors
        for neighbor, weight in adjacency_list.get(current_node, []):
            distance = current_distance + weight
            if neighbor not in distances or distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    # Reconstruct the path and return it
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = previous_nodes[node]
    path.reverse()

    return path, distances.get(goal, float('inf'))


# D* Lite algorithm (simplified version for dynamic path planning)
def d_star_lite(adjacency_list, start, goal, coordinates):
    # Use a variant of Dijkstra or A* that replans dynamically (simplified for now)
    return dijkstra(adjacency_list, start, goal, coordinates)


# Function to plan a path using D* or D* Lite
def path_planning_d_star(adjacency_list, goal_node, nodes_coordinates, algorithm="D*"):
    start_node = '1'  # Assuming start node is '1', can be modified to be dynamic

    if algorithm == "D*":
        path, distance = dijkstra(adjacency_list, start_node, goal_node, nodes_coordinates)
    elif algorithm == "D* Lite":
        path, distance = d_star_lite(adjacency_list, start_node, goal_node, nodes_coordinates)
    else:
        raise ValueError("Unknown algorithm specified")

    return path, distance


# Example adjacency list and node coordinates
adjacency_list = {
    '1': [('2', 1.2), ('3', 3.4), ('4', 7.9)],
    '2': [('4', 5.5)],
    '3': [('4', 12.6)]
}

nodes_coordinates = {
    '1': (0, 0),
    '2': (1, 1),
    '3': (2, 2),
    '4': (4, 4)
}

# Example usage:
goal_node = '2'
path, distance = path_planning_d_star(adjacency_list, goal_node, nodes_coordinates, algorithm="D* Lite")
print(f"Path: {path}, Distance: {distance}")