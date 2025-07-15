import osmnx as ox
import networkx as nx
import json
import random
import pickle
import pandas as pd


# ==============================================================================
#  Module 1: Feature Discovery Function (with English output)
# ==============================================================================

def discover_features_gdf(
        tags: dict,
        max_category_list=None,
        place_name="Singapore",

):
    """
    Discovers features for each specified category, applies a limit to each, and merges them.

    :param tags: A dictionary where keys are category names and values are OSM tags.
    :param place_name: The name of the area to query.
    :param max_category_list: A list of max items per category. A value of -1 means no limit.
    :return: A merged, filtered, and deduplicated GeoDataFrame with a 'category' column.
    """
    print(f"--- (Multi-category discovery) Processing categories with limits: {max_category_list} ---")
    all_gdfs = []  # To store the GeoDataFrame for each category
    i = 0

    # --- Step 1: Iterate through each defined category ---
    for category_name, tag_values in tags.items():
        print(f"\n--- Processing category: '{category_name}' (Tags: {tag_values}) ---")
        max_per_category = max_category_list[i]
        try:
            # Query each category individually
            features_gdf = ox.features_from_place(place_name, tag_values)

            if features_gdf.empty:
                print(f"Category '{category_name}' returned no features.")
                i += 1
                continue

                # Clean the data
            features_gdf.dropna(subset=['name'], inplace=True)
            features_gdf.drop_duplicates(subset='name', keep='first', inplace=True)

            # **GOAL 2: Add category information to each feature**
            features_gdf['category'] = category_name

            print(f"Found {len(features_gdf)} unique features for '{category_name}'.")

            # A limit of -1 means we should take all found features
            effective_limit = max_per_category if max_per_category > 0 else len(features_gdf)

            # Sort by Wikidata to prioritize more important places
            if 'wikidata' in features_gdf.columns:
                features_gdf.sort_values(by='wikidata', inplace=True, na_position='last', ascending=True)

                # Apply the limit
            if len(features_gdf) > effective_limit:
                print(
                    f"Trimming '{category_name}' from {len(features_gdf)} to the {effective_limit} most important ones.")
                features_gdf = features_gdf.head(effective_limit)

            all_gdfs.append(features_gdf)
            i += 1

        except Exception as e:
            print(f"An error occurred while processing category '{category_name}': {e}")
            i += 1

    # --- Step 2: Combine all results ---
    if not all_gdfs:
        print("No valid data was found for any category. Returning an empty GeoDataFrame.")
        return pd.DataFrame()

    print("\n--- Combining results from all categories... ---")
    final_gdf = pd.concat(all_gdfs, ignore_index=True)

    # --- Step 3: Final deduplication ---
    # A location might belong to multiple categories. This ensures each location is unique.
    print(f"Combined GDF has {len(final_gdf)} entries. Performing final deduplication by name...")
    final_gdf.drop_duplicates(subset='name', keep='first', inplace=True)
    final_gdf.reset_index(drop=True, inplace=True)

    print(f"Discovery complete! Returning a total of {len(final_gdf)} unique, important locations.")
    return final_gdf


# ==============================================================================
#  Module 2: Generic Graph Builder and Saver (with English output & category)
# ==============================================================================

def build_and_save_graph(
        locations_gdf: pd.DataFrame,
        output_name: str,
        place_name="Singapore",
        distance_threshold_km=3.0,
):
    """
    Receives a GeoDataFrame, builds a graph with categories, and saves it to files.
    """
    json_path = f"{output_name}_graph.json"
    vis_pickle = f"{output_name}_graph_for_visualization.pkl"
    print(f"\n--- (Graph Builder) Starting to build graph for '{output_name}'... ---")

    # --- Step 1: Download road network ---
    print("Step 1: Downloading road network...")
    G_road = ox.graph_from_place(place_name, network_type='drive', simplify=True)

    # --- Step 2: Match locations to nearest road nodes ---
    print("Step 2: Finding nearest road nodes for each location...")
    centroids = locations_gdf.geometry.centroid
    nearest_nodes = ox.nearest_nodes(G_road, X=centroids.x, Y=centroids.y)
    locations_gdf['node'] = nearest_nodes

    name2node = dict(zip(locations_gdf['name'], locations_gdf['node']))
    node2name = {v: k for k, v in name2node.items()}

    # --- Step 3: Build network and calculate distances ---
    print(f"Step 3: Calculating distances for {len(locations_gdf)} locations (threshold: {distance_threshold_km}km)...")
    records = []
    viz_paths = {}

    # The locations_gdf now contains the 'category' column
    for _, row in locations_gdf.iterrows():
        n1_name = row['name']
        n1_node = row['node']
        neighbors = {}
        try:
            lengths = nx.shortest_path_length(G_road, source=n1_node, weight='length')
            paths = nx.shortest_path(G_road, source=n1_node, weight='length')
        except nx.NetworkXNoPath:
            continue

        for target_node, length_m in lengths.items():
            if target_node not in node2name or n1_node == target_node:
                continue

            length_km = length_m / 1000
            if length_km <= distance_threshold_km:
                n2_name = node2name[target_node]
                neighbors[n2_name] = round(length_km, 3)
                key = tuple(sorted([n1_name, n2_name]))
                if key not in viz_paths:
                    viz_paths[key] = paths[target_node]

                    # **GOAL 2: Add the category to the final JSON record**
        rec = {
            "name": n1_name,
            "category": row['category'],
            "neighbors": neighbors,
            "reward": random.randint(50, 200)
        }
        records.append(rec)

        # --- Step 4: Save files ---
    print("Step 4: Saving results to files...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  -> Successfully wrote graph data to {json_path}")

    vis_obj = {"G": G_road, "gdf": locations_gdf, "viz_paths": viz_paths}
    with open(vis_pickle, 'wb') as f:
        pickle.dump(vis_obj, f)
    print(f"  -> Visualization data saved to {vis_pickle}")
    print(f"--- Build complete for '{output_name}' ---")


# ==============================================================================
#  Module 3: Main Execution Logic
# ==============================================================================

if __name__ == '__main__':
    # tags = {
    #     "major_parks": {'leisure': 'park'},
    #     "museums": {'tourism': 'museum'},
    #     "libraries": {'amenity': 'library'},
    #     "sports_venues": {'leisure': ['stadium', 'sports_centre']},
    #     "tourist_attractions": {'tourism': ['attraction', 'viewpoint']},
    #     "airports": {'aeroway': 'aerodrome'},
    #     "universities": {'amenity': 'university'},
    #     "shopping_malls": {'shop': 'mall'}
    # }
    # #
    # # # -1 means no limit (fetch all available)
    # tags_num = [-1, 20, 20, 20, 200, -1, -1,100]

    tags = {
        "major_parks": {'leisure': 'park'},
    }
    #
    # # -1 means no limit (fetch all available)
    tags_num = [10]

    # Call the discovery function
    poi_gdf = discover_features_gdf(tags=tags, max_category_list=tags_num)

    # Build the graph if data was found
    if not poi_gdf.empty:
        build_and_save_graph(
            locations_gdf=poi_gdf,
            output_name="TEST",
            distance_threshold_km=3.0
        )
    else:
        print("\nGraph building cancelled because no locations were found.")
