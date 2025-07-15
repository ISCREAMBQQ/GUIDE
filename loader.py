from typing import List, Optional
import pickle
import os
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from  ShortestPath import *
# --------------------------------------------------------------------------- #
# 主函数
# --------------------------------------------------------------------------- #
def visualize_path(
    start_name: str,
    end_name: str,
    vis_pickle_path: str,
    via_names: Optional[List[str]] = None,
    font_path: str = None,
    export_path: str = None,
    build_folium: bool = True,
    folium_html: str = "route_map.html"
):
    """
    规划并可视化“起点 → 途经点 → 终点”的最短路线。

    Parameters
    ----------
    start_name, end_name : str
        起点和终点名称，需出现在 pickle 中的 gdf['name'] 字段
    vis_pickle_path : str
        pickle 文件，需要包含 {'G': Graph, 'gdf': GeoDataFrame}
    via_names : list[str], optional
        途经点名称列表（顺序遵循此列表）
    font_path : str, optional
        matplotlib 中文字体路径
    export_path : str, optional
        节点 / 边 / 分段信息导出的 CSV 或 XLSX 路径
    build_folium : bool, default True
        是否构建交互式 Folium 地图
    folium_html : str, default "route_map.html"
        Folium 地图输出文件名
    """
    # ---------- Step 0 读取 pickle ----------
    print(f"[0] 载入数据：{vis_pickle_path}")
    with open(vis_pickle_path, "rb") as f:
        obj = pickle.load(f)

    G: nx.MultiDiGraph = obj["G"]
    gdf: gpd.GeoDataFrame = obj["gdf"]
    name_to_node = dict(zip(gdf["name"], gdf["node"]))

    # ---------- Step 1 名称检查 ----------
    via_names = via_names or []
    stop_names = [start_name] + via_names + [end_name]
    print(name_to_node)

    def check(name: str) -> int:
        if name not in name_to_node:
            raise ValueError(f"❌ 名称 '{name}' 未在数据集中找到，请检查拼写。")
        return name_to_node[name]

    stop_nodes = [check(n) for n in stop_names]

    # ---------- Step 2 分段最短路径 ----------
    print(f"[1] 规划分段最短路线，共 {len(stop_nodes) - 1} 段 ...")
    full_route = []
    leg_stats = []

    for i in range(len(stop_nodes) - 1):
        u, v = stop_nodes[i], stop_nodes[i + 1]
        u_name, v_name = stop_names[i], stop_names[i + 1]

        route_part = nx.shortest_path(G, u, v, weight="length")
        dist_m = nx.shortest_path_length(G, u, v, weight="length")

        leg_stats.append({
            "leg": f"{u_name} → {v_name}",
            "meters": dist_m,
            "kilometers": round(dist_m / 1000, 3)
        })

        full_route.extend(route_part if i == 0 else route_part[1:])

    total_m = sum(l["meters"] for l in leg_stats)
    print(f"    ✓ 总距离：{total_m/1000:.2f} km")

    # ---------- Step 3 生成节点 / 边 明细 ----------
    nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False, node_geometry=False)[["x", "y"]]
    route_nodes = nodes_gdf.loc[full_route].copy()
    route_nodes["seq"] = range(len(route_nodes))
    route_nodes["type"] = "waypoint"

    for name, node in zip(stop_names, stop_nodes):
        route_nodes.loc[node, "type"] = "stop"
        route_nodes.loc[node, "poi_name"] = name

    edge_rows = []
    for u, v in zip(full_route[:-1], full_route[1:]):
        data = G.get_edge_data(u, v, 0)
        edge_rows.append({
            "from_seq": route_nodes.loc[u, "seq"],
            "to_seq":   route_nodes.loc[v, "seq"],
            "from_node": u,
            "to_node": v,
            "street": data.get("name"),
            "length_m": data.get("length"),
            "highway": data.get("highway")
        })
    edges_df = pd.DataFrame(edge_rows)

    # ---------- Step 4 可选导出 ----------
    if export_path:
        print(f"[2] 导出数据 → {export_path}")
        if export_path.lower().endswith((".xls", ".xlsx")):
            with pd.ExcelWriter(export_path) as writer:
                route_nodes.to_excel(writer, sheet_name="nodes")
                edges_df.to_excel(writer, sheet_name="edges", index=False)
                pd.DataFrame(leg_stats).to_excel(writer, sheet_name="legs", index=False)
        else:  # CSV
            route_nodes.reset_index().to_csv(export_path, index=False)
            edges_df.to_csv(export_path, mode="a", index=False)
            pd.DataFrame(leg_stats).to_csv(export_path, mode="a", index=False)
        print("    ✓ 导出完成")

    # ---------- Step 5 Matplotlib 静态图 ----------
    print("[3] 绘制 Matplotlib 路线图 ...")
    fig, ax = ox.plot_graph_route(
        G, full_route,
        route_color="red", route_linewidth=3,
        node_size=0,
        edge_color="#CCCCCC",
        bgcolor="white",
        show=False, close=False
    )

    ax.scatter(route_nodes["x"], route_nodes["y"],
               c=route_nodes["type"].map({"stop": "magenta", "waypoint": "cyan"}),
               s=40, zorder=5)

    fp = fm.FontProperties(fname=font_path) if font_path else None
    for node, row in route_nodes[route_nodes["type"] == "stop"].iterrows():
        ax.text(row["x"], row["y"], row["poi_name"], fontproperties=fp,
                fontsize=10, weight="bold", color="black", zorder=6)

    title = f"Shortest Route: {' → '.join(stop_names)}\nTotal: {total_m/1000:.2f} km"
    ax.set_title(title, fontproperties=fp, fontsize=14)
    plt.show()

    # ---------- Step 6 Folium 交互地图 ----------
    if build_folium:
        print("[4] 构建 Folium 交互地图 ...")
        # osmnx 1.x 起 folium 工具在子模块 osmnx.folium
        ox_folium = ox.folium if hasattr(ox, "folium") else __import__("osmnx.folium", fromlist=["plot_graph_folium"])

        m = ox_folium.plot_graph_folium(
            G, tiles="cartodbpositron",
            graph_kwargs={'color': 'gray', 'weight': 1, 'opacity': 0.6}
        )
        m = ox_folium.plot_route_folium(
            G, full_route, route_map=m,
            color="#FF0000", weight=5, opacity=0.9
        )

        # 标注 stop 点
        import folium
        for node, row in route_nodes[route_nodes["type"] == "stop"].iterrows():
            folium.Marker(
                location=[row["y"], row["x"]],
                popup=row["poi_name"],
                icon=folium.Icon(color="purple", icon="flag")
            ).add_to(m)

        m.save(folium_html)
        print(f"    ✓ Folium 地图已保存为 {folium_html}")

    # ---------- Step 7 控制台摘要 ----------
    print("\n[5] 分段距离：")
    for leg in leg_stats:
        print(f"    · {leg['leg']:<30} {leg['kilometers']:.3f} km")

    print("√ 路径规划完毕\n")


# --------------------------------------------------------------------------- #
# CLI 示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # FONT_PATH = "/mnt/data/file-ngwyeoEN29l1M3O1QpdxCwkj-sider-font.ttf"  # 如需中文显示
    json_file_path = "Graph/singapore_categorized_pois_graph_updated.json"

    # 2. Replace "StartNodeName" with the name of your starting node.
    start_node_name = "National University of Singapore"

    # 3. Replace "EndNodeName" with the name of your ending node.
    end_node_name = "Changi Airport"

    # Run the calculation
    nodes = run_calculation_from_file(json_file_path, start_node_name, end_node_name)
    visualize_path(
        start_name=start_node_name,
        end_name=end_node_name,
        via_names=nodes,
        vis_pickle_path="Graph/singapore_categorized_pois_graph_for_visualization.pkl",
        export_path="route_details.xlsx",
        build_folium=True,
        folium_html="singapore_route.html"
    )