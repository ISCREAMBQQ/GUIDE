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
    name: str
    neighbors: Dict[str, Dict[str, float]]
    reward: int


class PathFinder:
    def __init__(self, graph_file: str):
        self.graph = {}
        self.all_nodes = set()
        self.shortest_path_cache = {}
        self.load_graph_from_file(graph_file)

    def load_graph_from_file(self, filename: str):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading or parsing {filename}: {e}")
            raise

        self.all_nodes = {item['name'] for item in data if 'name' in item}
        for item in data:
            self.all_nodes.update(item.get('neighbors', {}).keys())

        for node_name in self.all_nodes:
            self.graph[node_name] = Node(name=node_name, neighbors={}, reward=0)

        for item in data:
            node_name = item.get('name')
            if node_name not in self.graph:
                continue

            self.graph[node_name].reward = item.get('reward', 0)
            actual_distances = item.get('neighbors', {})
            reward_info = item.get('neighbors_reward', {})

            for neighbor, distance in actual_distances.items():
                if neighbor in reward_info and 'adjusted_value' in reward_info[neighbor]:
                    self.graph[node_name].neighbors[neighbor] = {
                        "actual_distance": distance,
                        "adjusted_value": reward_info[neighbor]['adjusted_value']
                    }

                    # Ensure graph is undirected
        for node_name, node_obj in list(self.graph.items()):
            for neighbor_name, weights in list(node_obj.neighbors.items()):
                if neighbor_name in self.graph and node_name not in self.graph[neighbor_name].neighbors:
                    self.graph[neighbor_name].neighbors[node_name] = weights

    def dijkstra_shortest_path(self, start: str, end: str, avoid_nodes: set = None,
                               weight_key: str = 'adjusted_value') -> Tuple[list, float]:
        cache_key = (start, end, frozenset(avoid_nodes) if avoid_nodes else None, weight_key)
        if cache_key in self.shortest_path_cache:
            return self.shortest_path_cache[cache_key]

        if start not in self.graph or end not in self.graph: return [], float('inf')

        nodes_to_avoid = avoid_nodes if avoid_nodes is not None else set()
        distances = {node: float('inf') for node in self.graph}
        predecessors = {node: None for node in self.graph}
        distances[start] = 0
        heap = [(0, start)]
        visited = set()

        while heap:
            current_dist, current_node = heapq.heappop(heap)
            if current_node in visited: continue
            visited.add(current_node)
            if current_node == end: break

            for neighbor, weights in self.graph[current_node].neighbors.items():
                if neighbor in visited or (neighbor in nodes_to_avoid and neighbor != end): continue
                weight = weights.get(weight_key)
                if weight is None: continue

                new_dist = current_dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current_node
                    heapq.heappush(heap, (new_dist, neighbor))

        if distances[end] == float('inf'): return [], float('inf')

        path = []
        current = end
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()

        result = (path, distances[end])
        self.shortest_path_cache[cache_key] = result
        return result

    def shortest_path(self, start: str, end: str, avoid_nodes: set = None, weight_key: str = 'adjusted_value') -> Tuple[
        list, float]:
        return self.dijkstra_shortest_path(start, end, avoid_nodes=avoid_nodes, weight_key=weight_key)

    def calculate_total_distance(self, start: str, waypoint_order: List[str], end: str, weight_key: str) -> float:
        total_distance = 0
        current_pos = start
        path = [start] + waypoint_order + [end]
        for i in range(len(path) - 1):
            _, distance = self.shortest_path(path[i], path[i + 1], weight_key=weight_key)
            if distance == float('inf'): return float('inf')
            total_distance += distance
        return total_distance

    def find_best_waypoint_order(self, start: str, waypoints: List[str], end: str) -> Tuple[List[str], float]:
        """Finds the best order for a given list of waypoints using adjusted_value."""
        if not waypoints:
            cost = self.calculate_total_distance(start, [], end, 'adjusted_value')
            return [], cost

        if len(waypoints) <= 7:  # Use permutations for smaller sets
            best_order_tuple = min(permutations(waypoints),
                                   key=lambda order: self.calculate_total_distance(start, list(order), end,
                                                                                   'actual_distance'))
            best_order = list(best_order_tuple)
        else:  # Use heuristics for larger sets
            # Using improved greedy + 2-opt as the heuristic
            order, _ = self.improved_greedy_with_end_consideration(start, waypoints, end)
            if not order: return [], float('inf')
            best_order, _ = self.two_opt_optimization(order, start, end)

        final_cost = self.calculate_total_distance(start, best_order, end, 'actual_distance')
        return best_order, final_cost

    def improved_greedy_with_end_consideration(self, start: str, waypoints: List[str], end: str) -> Tuple[
        List[str], float]:
        """Helper for ordering: Improved greedy using 'adjusted_value'."""
        remaining = set(waypoints)
        current_pos = start
        order = []
        while remaining:
            best_wp = min(remaining,
                          key=lambda wp: self.shortest_path(current_pos, wp, weight_key='adjusted_value')[1] + 0.3 *
                                         self.shortest_path(wp, end, weight_key='adjusted_value')[1], default=None)
            if best_wp is None: return [], float('inf')
            order.append(best_wp)
            remaining.remove(best_wp)
            current_pos = best_wp
        cost = self.calculate_total_distance(start, order, end, 'adjusted_value')
        return order, cost

    def two_opt_optimization(self, waypoint_order: List[str], start: str, end: str) -> Tuple[List[str], float]:
        """Helper for ordering: 2-opt using 'adjusted_value'."""
        best_cost = self.calculate_total_distance(start, waypoint_order, end, 'adjusted_value')
        order = waypoint_order[:]
        improved = True
        while improved:
            improved = False
            for i, j in permutations(range(len(order)), 2):
                if i >= j: continue
                new_order = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                new_cost = self.calculate_total_distance(start, new_order, end, 'adjusted_value')
                if new_cost < best_cost:
                    order = new_order
                    best_cost = new_cost
                    improved = True
                    break
            if improved: break
        return order, best_cost

    def find_path_with_waypoints(self, start: str, user_waypoints: List[str], end: str) -> Dict:
        """
        Implements the new 2-stage logic:
        1. Discover reward-based path to define all nodes to visit.
        2. Plan the final physical path through these nodes.
        """
        # --- 核心逻辑 Stage 1: 发现高价值节点 ---
        # 执行一次基于'adjusted_value'的迪杰斯特拉，找出“奖励”路径
        discovered_path_nodes, _ = self.shortest_path(start, end, weight_key='adjusted_value')

        if not discovered_path_nodes:
            # 如果连奖励路径都找不到，直接返回失败
            return {"path": [], "actual_distance": float('inf'), "waypoint_order": []}

        # --- 核心逻辑 Stage 2: 整合并排序所有必经点 ---
        # 将用户指定的waypoints和我们发现的节点合并，并去重
        all_intermediate_nodes = set(user_waypoints) | set(discovered_path_nodes)
        # 移除起点和终点，因为它们是路径的边界，而不是中间点
        all_intermediate_nodes.discard(start)
        all_intermediate_nodes.discard(end)

        final_waypoints = list(all_intermediate_nodes)

        # 为这个最终的节点列表，找到基于'adjusted_value'的最佳访问顺序
        best_order, _ = self.find_best_waypoint_order(start, final_waypoints, end)

        # --- 核心逻辑 Stage 3: 生成最终物理路径 ---
        # 使用 'actual_distance' 来连接 'best_order' 中的点，生成最终路径
        complete_path = []
        visited_nodes = set()
        total_actual_distance = 0
        path_segments = [start] + best_order + [end]

        for i in range(len(path_segments) - 1):
            current_pos, next_pos = path_segments[i], path_segments[i + 1]
            if i == 0:
                complete_path.append(current_pos)
                visited_nodes.add(current_pos)

            # 使用 'actual_distance' 生成无回头路的物理路径段
            segment_path, segment_distance = self.shortest_path(
                current_pos, next_pos, avoid_nodes=visited_nodes - {current_pos}, weight_key='actual_distance'
            )

            if not segment_path:
                print(f"CRITICAL ERROR: Cannot find a non-repeating physical path from {current_pos} to {next_pos}.")
                return {"path": [], "actual_distance": float('inf'), "waypoint_order": best_order}

            new_nodes = segment_path[1:]
            complete_path.extend(new_nodes)
            visited_nodes.update(new_nodes)
            total_actual_distance += segment_distance

        final_adjusted_distance = self.calculate_total_distance(start, best_order, end, 'adjusted_value')

        return {
            "path": complete_path, "adjusted_distance": final_adjusted_distance,
            "actual_distance": total_actual_distance, "waypoint_order": best_order
        }


def find_path(graph_file: str, start: str, waypoints: List[str], end: str, objective: str = "distance") -> Dict:
    try:
        path_finder = PathFinder(graph_file)
        result = path_finder.find_path_with_waypoints(start, waypoints, end)
        path = result.get("path")

        if not path:
            return {"success": False, "message": "No path found", **result}

        total_reward = sum(
            path_finder.graph[node_name].reward for node_name in set(path) if node_name in path_finder.graph)

        return {
            "success": True, "path": path,
            "total_adjusted_distance": result.get("adjusted_distance"),
            "total_actual_distance": result.get("actual_distance"),
            "total_reward": total_reward, "waypoint_order": result.get("waypoint_order")
        }
    except Exception as e:
        import traceback
        return {
            "success": False, "message": f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}",
            "path": [], "total_adjusted_distance": float('inf'), "total_actual_distance": float('inf'),
            "total_reward": 0, "waypoint_order": []
        }
