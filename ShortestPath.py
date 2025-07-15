import json


def find_shortest_path(graph_data, start_node_name, end_node_name):
    """
    Calculates the shortest path in a graph with negative edge weights
    using the Bellman-Ford algorithm.

    Args:
        graph_data (list): A list of dictionaries, where each dictionary represents a node.
        start_node_name (str): The name of the starting node.
        end_node_name (str): The name of the ending node.

    Returns:
        tuple: A tuple containing the shortest path distance (float) and the path
               as a list of node names (list[str]).
               Returns (float('inf'), []) if no path exists.
               Raises a ValueError if a negative-weight cycle is detected.
    """
    # Create a mapping from node names to their full data for easy access
    graph = {node['name']: node for node in graph_data}
    nodes = list(graph.keys())

    if start_node_name not in nodes or end_node_name not in nodes:
        raise ValueError("Start or end node not in the graph")

    # Step 1: Initialize distances and predecessors
    distances = {node: float('inf') for node in nodes}
    predecessors = {node: None for node in nodes}
    distances[start_node_name] = 0

    # Step 2: Relax edges repeatedly (V-1 times)
    for i in range(len(nodes) - 1):
        for node_name in nodes:
            if 'neighbors' in graph[node_name]:
                for neighbor, property in graph[node_name]['neighbors_reward'].items():
                    weight = property['adjusted_value']
                    if neighbor in distances and distances[node_name] != float('inf') and \
                            distances[node_name] + weight < distances[neighbor]:
                        distances[neighbor] = distances[node_name] + weight
                        predecessors[neighbor] = node_name

    # Step 3: Check for negative-weight cycles
    for node_name in nodes:
        if 'neighbors_reward' in graph[node_name]:
            for neighbor, property in graph[node_name]['neighbors_reward'].items():
                weight = property['adjusted_value']
                if neighbor in distances and distances[node_name] != float('inf') and \
                        distances[node_name] + weight < distances[neighbor]:
                    raise ValueError("Graph contains a negative-weight cycle")

    # Step 4: Reconstruct the path if a path exists
    path = []
    current_node = end_node_name
    if distances[end_node_name] == float('inf'):
        return float('inf'), []

    while current_node is not None:
        path.insert(0, current_node)
        current_node = predecessors.get(current_node)

    if path and path[0] == start_node_name:
        return distances[end_node_name], path
    else:
        return float('inf'), []


def run_calculation_from_file(file_path, start_node, end_node):
    """
    Loads a graph from a JSON file and calculates the shortest path.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        shortest_distance, path_nodes = find_shortest_path(graph_data, start_node, end_node)

        if not path_nodes:
            print(f"No path found from '{start_node}' to '{end_node}'.")
        else:
            print(f"Shortest path from '{start_node}' to '{end_node}':")
            print(f"  -> Distance: {shortest_distance:.4f}")
            print(f"  -> Path: {' -> '.join(path_nodes)}")
        print(132132,'assada',path_nodes)
        return path_nodes

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{file_path}'. Check the file format.")
    except ValueError as e:
        print(f"Error: {e}")
    except KeyError as e:
        print(f"Error: Node {e} not found in the graph's neighbor list. Check your JSON data for consistency.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- HOW TO USE ---
if __name__ == "__main__":
    # 1. Replace "your_graph_file.json" with the actual path to your JSON file.
    #    Make sure the script and the JSON file are in the same directory,
    #    or provide the full path to the file.
    json_file_path = "singapore_categorized_pois_graph_with_latlng.json"

    # 2. Replace "StartNodeName" with the name of your starting node.
    start_node_name = "Redhill"

    # 3. Replace "EndNodeName" with the name of your ending node.
    end_node_name = ""

    # Run the calculation
    nodes = run_calculation_from_file(json_file_path, start_node_name, end_node_name)
    print(nodes)
