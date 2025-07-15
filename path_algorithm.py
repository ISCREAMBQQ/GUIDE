import json
import os
import heapq
import datetime
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import permutations


@dataclass
class Node:
    """Node in the graph"""
    name: str
    neighbors: Dict[str, float]
    reward: int


class PathFinder:
    def __init__(self, graph_file: str):
        """Initialize path finder"""
        self.graph = {}
        self.all_nodes = set()
        self.shortest_path_cache = {}  # 缓存最短路径结果
        self.load_graph_from_file(graph_file)

    def load_graph_from_file(self, filename: str):
        """Load graph data from JSON file, using neighbors_reward[neighbor]['adjusted_value'] as edge weight"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"File {filename} does not exist")
            raise
        except json.JSONDecodeError:
            print(f"JSON file format error: {filename}")
            raise

        # Initialize graph
        self.graph = {}
        self.all_nodes = set()

        # Collect all nodes
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'name' in item:
                    node_name = item['name']
                    self.all_nodes.add(node_name)
                    # Add neighbors from the neighbors_reward field
                    for neighbor in item.get('neighbors_reward', {}):
                        self.all_nodes.add(neighbor)

        # Create Node objects
        for node_name in self.all_nodes:
            self.graph[node_name] = Node(name=node_name, neighbors={}, reward=0)

        # Fill data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'name' in item:
                    node_name = item['name']
                    if node_name in self.graph:
                        # Use neighbors_reward for distances
                        neighbors_reward = item.get('neighbors_reward', {})
                        neighbors = {}
                        for neighbor, info in neighbors_reward.items():
                            adjusted_value = info.get('adjusted_value', None)
                            if adjusted_value is not None:
                                neighbors[neighbor] = adjusted_value
                        self.graph[node_name].neighbors = neighbors
                        # Use the reward field for node rewards
                        self.graph[node_name].reward = item.get('reward', 0)

        # Make undirected - ensure bidirectional connections
        for node_name, node_obj in list(self.graph.items()):
            for neighbor_name, distance in list(node_obj.neighbors.items()):
                if neighbor_name in self.graph:
                    if node_name not in self.graph[neighbor_name].neighbors:
                        self.graph[neighbor_name].neighbors[node_name] = distance

    def dijkstra_shortest_path(self, start: str, end: str) -> Tuple[list, float]:
        """Dijkstra's algorithm for shortest path (non-negative weights only)"""
        if start not in self.graph or end not in self.graph:
            return [], float('inf')
        distances = {node: float('inf') for node in self.graph}
        predecessors = {node: None for node in self.graph}
        distances[start] = 0
        heap = [(0, start)]
        visited = set()
        while heap:
            current_dist, current_node = heapq.heappop(heap)
            if current_node in visited:
                continue
            visited.add(current_node)
            if current_node == end:
                break
            for neighbor, weight in self.graph[current_node].neighbors.items():
                if neighbor in visited:
                    continue
                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(heap, (new_dist, neighbor))
        if distances[end] == float('inf'):
            return [], float('inf')
        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        return path, distances[end]

    def shortest_path(self, start: str, end: str) -> Tuple[list, float]:
        """Use Dijkstra for shortest path query"""
        return self.dijkstra_shortest_path(start, end)

    def calculate_total_distance(self, start: str, waypoint_order: List[str], end: str) -> float:
        """Calculate total distance for a given waypoint order"""
        total_distance = 0
        current_pos = start

        for waypoint in waypoint_order:
            _, distance = self.shortest_path(current_pos, waypoint)
            if distance == float('inf'):
                return float('inf')
            total_distance += distance
            current_pos = waypoint

        # Add distance to end
        _, final_distance = self.shortest_path(current_pos, end)
        if final_distance == float('inf'):
            return float('inf')

        return total_distance + final_distance

    def greedy_nearest_neighbor(self, start: str, waypoints: List[str], end: str) -> Tuple[List[str], float]:
        """Basic greedy nearest neighbor algorithm"""
        remaining_waypoints = set(waypoints)
        current_pos = start
        waypoint_order = []

        while remaining_waypoints:
            nearest_waypoint = None
            min_distance = float('inf')

            for waypoint in remaining_waypoints:
                _, distance = self.shortest_path(current_pos, waypoint)
                if distance < min_distance:
                    min_distance = distance
                    nearest_waypoint = waypoint

            if nearest_waypoint is None:
                return [], float('inf')

            waypoint_order.append(nearest_waypoint)
            remaining_waypoints.remove(nearest_waypoint)
            current_pos = nearest_waypoint

        return waypoint_order, self.calculate_total_distance(start, waypoint_order, end)

    def improved_greedy_with_end_consideration(self, start: str, waypoints: List[str], end: str) -> Tuple[
        List[str], float]:
        """Improved greedy algorithm that considers distance to end point"""
        remaining_waypoints = set(waypoints)
        current_pos = start
        waypoint_order = []

        while remaining_waypoints:
            best_waypoint = None
            best_score = float('inf')

            for waypoint in remaining_waypoints:
                # Distance to current waypoint
                _, distance_to_waypoint = self.shortest_path(current_pos, waypoint)
                if distance_to_waypoint == float('inf'):
                    continue

                # Distance from waypoint to end
                _, distance_to_end = self.shortest_path(waypoint, end)
                if distance_to_end == float('inf'):
                    continue

                # Score: balance between distance to waypoint and distance to end
                # Lower score is better
                score = distance_to_waypoint + 0.3 * distance_to_end

                if score < best_score:
                    best_score = score
                    best_waypoint = waypoint

            if best_waypoint is None:
                return [], float('inf')

            waypoint_order.append(best_waypoint)
            remaining_waypoints.remove(best_waypoint)
            current_pos = best_waypoint

        return waypoint_order, self.calculate_total_distance(start, waypoint_order, end)

    def multi_start_greedy(self, start: str, waypoints: List[str], end: str, num_starts: int = 5) -> Tuple[
        List[str], float]:
        """Multi-start greedy algorithm with different starting waypoints"""
        if len(waypoints) <= 1:
            return self.greedy_nearest_neighbor(start, waypoints, end)

        best_order = []
        best_distance = float('inf')

        # Try different starting waypoints
        for _ in range(min(num_starts, len(waypoints))):
            # Randomly select a starting waypoint
            start_waypoint = random.choice(list(waypoints))
            remaining_waypoints = set(waypoints) - {start_waypoint}

            # Build path starting from this waypoint
            current_pos = start
            waypoint_order = [start_waypoint]

            # Add remaining waypoints using greedy approach
            while remaining_waypoints:
                nearest_waypoint = None
                min_distance = float('inf')

                for waypoint in remaining_waypoints:
                    _, distance = self.shortest_path(current_pos, waypoint)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_waypoint = waypoint

                if nearest_waypoint is None:
                    break

                waypoint_order.append(nearest_waypoint)
                remaining_waypoints.remove(nearest_waypoint)
                current_pos = nearest_waypoint

            # Calculate total distance
            total_distance = self.calculate_total_distance(start, waypoint_order, end)

            if total_distance < best_distance:
                best_distance = total_distance
                best_order = waypoint_order

        return best_order, best_distance

    def two_opt_optimization(self, waypoint_order: List[str], start: str, end: str) -> Tuple[List[str], float]:
        """2-opt optimization to improve waypoint order"""
        if len(waypoint_order) <= 2:
            return waypoint_order, self.calculate_total_distance(start, waypoint_order, end)

        improved = True
        best_distance = self.calculate_total_distance(start, waypoint_order, end)

        while improved:
            improved = False

            for i in range(len(waypoint_order) - 1):
                for j in range(i + 2, len(waypoint_order)):
                    # Try 2-opt swap
                    new_order = waypoint_order[:i + 1] + waypoint_order[i + 1:j + 1][::-1] + waypoint_order[j + 1:]
                    new_distance = self.calculate_total_distance(start, new_order, end)

                    if new_distance < best_distance:
                        waypoint_order = new_order
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break

        return waypoint_order, best_distance

    def find_path_with_waypoints(self, start: str, waypoints: List[str], end: str) -> Tuple[
        List[str], float, List[str]]:
        """Find path with waypoints using improved algorithms optimized for ≤10 waypoints"""
        if not waypoints:
            path, distance = self.shortest_path(start, end)
            return path, distance, []

        # For small number of waypoints (≤4), try all permutations for exact solution
        if len(waypoints) <= 4:
            best_distance = float('inf')
            best_path = []
            best_order = []

            for waypoint_order in permutations(waypoints):
                current_distance = 0
                current_path = []
                current_pos = start
                valid_path = True

                for waypoint in waypoint_order:
                    segment_path, segment_distance = self.shortest_path(current_pos, waypoint)
                    if not segment_path or segment_distance == float('inf'):
                        valid_path = False
                        break

                    if current_path:
                        current_path.extend(segment_path[1:])
                    else:
                        current_path.extend(segment_path)
                    current_distance += segment_distance
                    current_pos = waypoint

                if not valid_path:
                    continue

                final_segment, final_distance = self.shortest_path(current_pos, end)
                if not final_segment or final_distance == float('inf'):
                    continue

                current_path.extend(final_segment[1:])
                current_distance += final_distance

                if current_distance < best_distance:
                    best_distance = current_distance
                    best_path = current_path
                    best_order = list(waypoint_order)

            return best_path, best_distance, best_order
        else:
            # For 5-10 waypoints, use optimized algorithm combination
            best_order = []
            best_distance = float('inf')

            # Optimized algorithm selection for 5-10 waypoints
            if len(waypoints) <= 7:
                # For 5-7 waypoints: try basic greedy and improved greedy
                algorithms = [
                    ("Basic Greedy", self.greedy_nearest_neighbor),
                    ("Improved Greedy", self.improved_greedy_with_end_consideration)
                ]
            else:
                # For 8-10 waypoints: add multi-start greedy with fewer iterations
                algorithms = [
                    ("Basic Greedy", self.greedy_nearest_neighbor),
                    ("Improved Greedy", self.improved_greedy_with_end_consideration),
                    ("Multi-start Greedy (3)", lambda s, w, e: self.multi_start_greedy(s, w, e, 3))
                ]

            for name, algorithm in algorithms:
                try:
                    order, distance = algorithm(start, waypoints, end)
                    if distance < best_distance:
                        best_distance = distance
                        best_order = order
                except Exception as e:
                    print(f"Algorithm {name} failed: {e}")
                    continue

            # Apply 2-opt optimization to the best result
            if best_order:
                optimized_order, optimized_distance = self.two_opt_optimization(best_order, start, end)
                if optimized_distance < best_distance:
                    best_order = optimized_order
                    best_distance = optimized_distance

            # Build complete path
            if not best_order:
                return [], float('inf'), []

            complete_path = []
            current_pos = start
            total_distance = 0

            for waypoint in best_order:
                segment_path, segment_distance = self.shortest_path(current_pos, waypoint)
                if not segment_path:
                    return [], float('inf'), []

                if complete_path:
                    complete_path.extend(segment_path[1:])
                else:
                    complete_path.extend(segment_path)
                total_distance += segment_distance
                current_pos = waypoint

            final_segment, final_distance = self.shortest_path(current_pos, end)
            if not final_segment:
                return [], float('inf'), []

            complete_path.extend(final_segment[1:])
            total_distance += final_distance

            return complete_path, total_distance, best_order


def find_path(graph_file: str, start: str, waypoints: List[str], end: str, objective: str = "distance") -> Dict:
    """Main function to find path with waypoints
    
    Args:
        graph_file: Path to the JSON graph file
        start: Starting node name
        waypoints: List of waypoint node names
        end: Ending node name
        objective: Optimization objective - "distance" or "reward"
    """
    try:
        path_finder = PathFinder(graph_file)

        if objective == "reward":
            # For reward maximization, we need to implement a different algorithm
            # For now, we'll use the same distance-based algorithm but calculate total reward
            path, total_distance, waypoint_order = path_finder.find_path_with_waypoints(start, waypoints, end)
        else:
            # Default distance minimization
            path, total_distance, waypoint_order = path_finder.find_path_with_waypoints(start, waypoints, end)

        if not path:
            return {
                "success": False,
                "message": "No path found",
                "path": [],
                "total_distance": float('inf'),
                "total_reward": 0,
                "waypoint_order": []
            }

        # Calculate total reward for the path
        total_reward = 0
        visited_nodes = set()
        for node_name in path:
            if node_name in path_finder.graph and node_name not in visited_nodes:
                total_reward += path_finder.graph[node_name].reward
                visited_nodes.add(node_name)

        return {
            "success": True,
            "path": path,
            "total_distance": total_distance,
            "total_reward": total_reward,
            "waypoint_order": waypoint_order
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "path": [],
            "total_distance": float('inf'),
            "total_reward": 0,
            "waypoint_order": []
        }


if __name__ == "__main__":
    # 修改为新的输入文件名
    json_file = "Graph/graph_with_updated_rewards_and_weights.json"

    try:
        # Test with multiple waypoints to see improvement
        waypoints = ["Paya Lebar", "Dhoby Ghaut", "Orchard"]
        result = find_path(json_file, "MacPherson", waypoints, "Serangoon", "distance")

        # Save result
        output_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "result": result
        }

        # Create results directory if it doesn't exist
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(results_dir, f"path_finding_{timestamp}.json")

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {output_filename}")

        if result["success"]:
            print(f"Path: {' -> '.join(result['path'])}")
            print(f"Distance: {result['total_distance']:.2f}")
            print(f"Waypoint order: {' -> '.join(result['waypoint_order'])}")
        else:
            print(f"Error: {result['message']}")

    except Exception as e:
        print(f"Runtime error: {str(e)}")
