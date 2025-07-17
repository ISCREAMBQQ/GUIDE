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
    neighbors: Dict[str, float]  # adjusted_value for path finding
    actual_distances: Dict[str, float]  # actual distances for reordering
    reward: int


class PathFinder:
    def __init__(self, graph_file: str):
        """Initialize path finder"""
        self.graph = {}
        self.all_nodes = set()
        self.shortest_path_cache = {}
        self.load_graph_from_file(graph_file)

    def load_graph_from_file(self, filename: str):
        """Load graph data from JSON file"""
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
                    for neighbor in item.get('neighbors_reward', {}):
                        self.all_nodes.add(neighbor)

        # Create Node objects
        for node_name in self.all_nodes:
            self.graph[node_name] = Node(
                name=node_name,
                neighbors={},
                actual_distances={},
                reward=0
            )

        # Fill data
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'name' in item:
                    node_name = item['name']
                    if node_name in self.graph:
                        neighbors_reward = item.get('neighbors_reward', {})
                        neighbors = {}
                        actual_distances = {}

                        for neighbor, info in neighbors_reward.items():
                            # 获取adjusted_value用于路径搜索
                            adjusted_value = info.get('adjusted_value', None)
                            if adjusted_value is not None:
                                neighbors[neighbor] = adjusted_value

                            # 获取actual_distance用于重新排序
                            actual_distance = info.get('actual_distance', adjusted_value)
                            if actual_distance is not None:
                                actual_distances[neighbor] = actual_distance

                        self.graph[node_name].neighbors = neighbors
                        self.graph[node_name].actual_distances = actual_distances
                        self.graph[node_name].reward = item.get('reward', 0)

        # Make undirected - ensure bidirectional connections
        for node_name, node_obj in list(self.graph.items()):
            for neighbor_name, distance in list(node_obj.neighbors.items()):
                if neighbor_name in self.graph:
                    if node_name not in self.graph[neighbor_name].neighbors:
                        self.graph[neighbor_name].neighbors[node_name] = distance
                        # 同时设置实际距离
                        actual_dist = node_obj.actual_distances.get(neighbor_name, distance)
                        self.graph[neighbor_name].actual_distances[node_name] = actual_dist

    def dijkstra_shortest_path_actual(self, start: str, end: str, nodes_to_visit: set = None) -> Tuple[list, float]:
        """使用实际距离的Dijkstra算法，只经过指定的节点集合"""
        if start not in self.graph or end not in self.graph:
            return [], float('inf')

        # 如果指定了要访问的节点集合，确保起点和终点都在其中
        if nodes_to_visit is not None:
            if start not in nodes_to_visit or end not in nodes_to_visit:
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

            for neighbor in self.graph[current_node].actual_distances:
                if neighbor in visited:
                    continue

                # 如果指定了节点集合，只考虑集合内的节点
                if nodes_to_visit is not None and neighbor not in nodes_to_visit:
                    continue

                # 使用实际距离
                weight = self.graph[current_node].actual_distances.get(neighbor, float('inf'))
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

    def reorder_path_nodes(self, original_path: List[str], start: str, waypoints: List[str], end: str) -> Tuple[
        List[str], float]:
        """
        重新排序路径中的所有节点，使用实际距离避免回头路

        Args:
            original_path: 原始路径（可能包含回头路）
            start: 起点
            waypoints: 必经点列表
            end: 终点

        Returns:
            重新排序后的路径和总距离
        """
        # 收集路径中的所有唯一节点
        path_nodes = set(original_path)

        # 确保必经点都在路径节点中
        required_nodes = set(waypoints) | {start, end}
        path_nodes.update(required_nodes)

        # 如果节点数量较少，尝试所有排列找最优解
        if len(waypoints) <= 4:
            best_path = []
            best_distance = float('inf')

            for perm in permutations(waypoints):
                # 构建完整路径：起点 -> 排列的必经点 -> 终点
                current_distance = 0
                current_path = [start]
                current_pos = start
                valid = True

                # 经过所有必经点
                for waypoint in perm:
                    segment_path, segment_dist = self.dijkstra_shortest_path_actual(
                        current_pos, waypoint, path_nodes
                    )
                    if not segment_path:
                        valid = False
                        break
                    current_path.extend(segment_path[1:])
                    current_distance += segment_dist
                    current_pos = waypoint

                if not valid:
                    continue

                # 到达终点
                final_path, final_dist = self.dijkstra_shortest_path_actual(
                    current_pos, end, path_nodes
                )
                if not final_path:
                    continue

                current_path.extend(final_path[1:])
                current_distance += final_dist

                if current_distance < best_distance:
                    best_distance = current_distance
                    best_path = current_path

            return best_path, best_distance

        else:
            # 对于较多必经点，使用启发式方法
            # 使用最近邻算法，但基于实际距离
            remaining_waypoints = set(waypoints)
            current_pos = start
            waypoint_order = []

            while remaining_waypoints:
                nearest_waypoint = None
                min_distance = float('inf')

                for waypoint in remaining_waypoints:
                    _, distance = self.dijkstra_shortest_path_actual(
                        current_pos, waypoint, path_nodes
                    )
                    if distance < min_distance:
                        min_distance = distance
                        nearest_waypoint = waypoint

                if nearest_waypoint is None:
                    break

                waypoint_order.append(nearest_waypoint)
                remaining_waypoints.remove(nearest_waypoint)
                current_pos = nearest_waypoint

            # 构建最终路径
            final_path = [start]
            current_pos = start
            total_distance = 0

            for waypoint in waypoint_order:
                segment_path, segment_dist = self.dijkstra_shortest_path_actual(
                    current_pos, waypoint, path_nodes
                )
                if segment_path:
                    final_path.extend(segment_path[1:])
                    total_distance += segment_dist
                    current_pos = waypoint

            # 到达终点
            end_path, end_dist = self.dijkstra_shortest_path_actual(
                current_pos, end, path_nodes
            )
            if end_path:
                final_path.extend(end_path[1:])
                total_distance += end_dist

            return final_path, total_distance

    def dijkstra_shortest_path(self, start: str, end: str, avoid_nodes: set = None) -> Tuple[list, float]:
        """原有的Dijkstra算法，使用adjusted_value"""
        # ... [保持原有实现不变] ...
        if start not in self.graph or end not in self.graph:
            return [], float('inf')

        nodes_to_avoid = avoid_nodes if avoid_nodes is not None else set()
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

                if neighbor in nodes_to_avoid and neighbor != end:
                    continue

                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(heap, (new_dist, neighbor))

        if distances[end] == float('inf'):
            return [], float('inf')

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        return path, distances[end]

    def find_path_with_waypoints(self, start: str, waypoints: List[str], end: str) -> Tuple[
        List[str], float, List[str]]:
        """改进的路径查找，包含重新排序步骤"""
        if not waypoints:
            path, distance = self.shortest_path(start, end)
            return path, distance, []

        # 首先使用原有算法找到包含所有必经点的初始路径
        initial_path, initial_distance, waypoint_order = self._find_initial_path(start, waypoints, end)

        if not initial_path:
            return [], float('inf'), []

        # 使用实际距离重新排序路径节点，避免回头路
        reordered_path, actual_distance = self.reorder_path_nodes(
            initial_path, start, waypoints, end
        )

        return reordered_path, actual_distance, waypoint_order

    def _find_initial_path(self, start: str, waypoints: List[str], end: str) -> Tuple[List[str], float, List[str]]:
        """原有的路径查找逻辑"""
        # [这里保持原有的find_path_with_waypoints实现]
        if len(waypoints) <= 4:
            best_distance = float('inf')
            best_path = []
            best_order = []

            for waypoint_order_tuple in permutations(waypoints):
                waypoint_order = list(waypoint_order_tuple)
                current_distance = 0
                current_path = [start]
                visited_nodes = {start}
                current_pos = start
                valid_path = True

                for waypoint in waypoint_order:
                    avoid_set = visited_nodes - {current_pos}
                    segment_path, segment_distance = self.shortest_path(current_pos, waypoint, avoid_nodes=avoid_set)

                    if not segment_path:
                        valid_path = False
                        break

                    current_path.extend(segment_path[1:])
                    visited_nodes.update(segment_path[1:])
                    current_distance += segment_distance
                    current_pos = waypoint

                if not valid_path:
                    continue

                avoid_set = visited_nodes - {current_pos}
                final_segment, final_distance = self.shortest_path(current_pos, end, avoid_nodes=avoid_set)
                if not final_segment:
                    continue

                current_path.extend(final_segment[1:])
                current_distance += final_distance

                if current_distance < best_distance:
                    best_distance = current_distance
                    best_path = current_path
                    best_order = waypoint_order

            return best_path, best_distance, best_order
        else:
            # 对于5-10个必经点的情况，使用启发式算法
            best_order = []
            best_distance = float('inf')

            if len(waypoints) <= 7:
                algorithms = [
                    ("Basic Greedy", self.greedy_nearest_neighbor),
                    ("Improved Greedy", self.improved_greedy_with_end_consideration)
                ]
            else:
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

            if best_order:
                optimized_order, _ = self.two_opt_optimization(best_order, start, end)
                best_order = optimized_order

            if not best_order:
                return [], float('inf'), []

            complete_path = [start]
            visited_nodes = {start}
            current_pos = start
            total_distance = 0

            for waypoint in best_order:
                avoid_set = visited_nodes - {current_pos}
                segment_path, segment_distance = self.shortest_path(current_pos, waypoint, avoid_nodes=avoid_set)

                if not segment_path:
                    return [], float('inf'), best_order

                complete_path.extend(segment_path[1:])
                visited_nodes.update(segment_path[1:])
                total_distance += segment_distance
                current_pos = waypoint

            avoid_set = visited_nodes - {current_pos}
            final_segment, final_distance = self.shortest_path(current_pos, end, avoid_nodes=avoid_set)

            if not final_segment:
                return [], float('inf'), best_order

            complete_path.extend(final_segment[1:])
            total_distance += final_distance

            return complete_path, total_distance, best_order

    # [保留所有其他原有方法，如shortest_path, calculate_total_distance等]
    def shortest_path(self, start: str, end: str, avoid_nodes: set = None) -> Tuple[list, float]:
        """Use Dijkstra for shortest path query, with an option to avoid certain nodes."""
        return self.dijkstra_shortest_path(start, end, avoid_nodes=avoid_nodes)

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
                _, distance_to_waypoint = self.shortest_path(current_pos, waypoint)
                if distance_to_waypoint == float('inf'):
                    continue

                _, distance_to_end = self.shortest_path(waypoint, end)
                if distance_to_end == float('inf'):
                    continue

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

        for _ in range(min(num_starts, len(waypoints))):
            start_waypoint = random.choice(list(waypoints))
            remaining_waypoints = set(waypoints) - {start_waypoint}
            current_pos = start
            waypoint_order = [start_waypoint]

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


def find_path(graph_file: str, start: str, waypoints: List[str], end: str, objective: str = "distance") -> Dict:
    """Main function to find path with waypoints"""
    try:
        path_finder = PathFinder(graph_file)

        if objective == "reward":
            path, total_distance, waypoint_order = path_finder.find_path_with_waypoints(start, waypoints, end)
        else:
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
