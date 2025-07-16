import streamlit as st
import json
import folium
from folium.plugins import MarkerCluster
import os
import networkx as nx
# --- Import custom user analysis and pathfinding modules ---
from UserAnalysis import SemanticSimilarityCalculator
from path_algorithm import find_path


# --- 1. Global Setup ---
BASE_GRAPH_FILE = "Graph/GUIDE_037.json"
STOPWORDS_FILE = "ENGLISH_STOP.txt"


# --- 2. Initialization and Cached Functions ---
@st.cache_data
def load_pristine_data(file_path):
    """Loads the base JSON data from the specified file path."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Critical error loading base data file '{file_path}': {e}")
        st.stop()


@st.cache_resource
def get_similarity_calculator():
    """Initializes and caches the semantic similarity calculator."""
    if not os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, "w") as f: f.write("want\nfind\nplace\ngo\nsee")
    return SemanticSimilarityCalculator(stopwords_file_path=STOPWORDS_FILE)


@st.cache_resource
def load_networkx_graph(_graph_data):
    """Loads graph data into a NetworkX object for efficient pathfinding."""
    G = nx.Graph()
    poi_map = {p['name']: p for p in _graph_data}
    for poi in _graph_data:
        G.add_node(poi['name'])
        if 'neighbors' in poi and isinstance(poi['neighbors'], dict):
            for neighbor_name, distance in poi['neighbors'].items():
                if neighbor_name in poi_map:
                    G.add_edge(poi['name'], neighbor_name, weight=distance)
    return G


# --- 3. Core State and NEW Waypoint Suggestion Function ---
def initialize_session_state():
    """Initializes session state variables on the first run of the app."""
    if 'app_initialized' not in st.session_state:
        st.session_state.waypoints = []
        st.session_state.calc_path = None
        st.session_state.suggestions = []  # To hold the top 3 suggestions
        st.session_state.app_initialized = True


def find_best_waypoint(_poi_data, start_pt, end_pt, user_demand):
    """
    Finds the best 3 waypoint suggestions based on user interest and path proximity.
    1. Calculates the optimal path between the start and end points.
    2. Gathers all nodes on the path and their direct neighbors into a candidate set.
    3. Calculates the semantic similarity between the user's demand and the 'feature' of each candidate.
    4. Returns the top 3 most relevant candidates.
    """
    if not user_demand.strip():
        st.toast("Please describe your interests first to get suggestions.", icon="ℹ️")
        return []

        # 1. Calculate the optimal path between start and end.
    path_result = find_path(BASE_GRAPH_FILE, start_pt, [], end_pt, "distance")
    base_path = path_result['path'] if path_result and 'path' in path_result else []

    # Create a mapping from name to the full POI data for efficient lookups
    poi_map = {p['name']: p for p in _poi_data}

    # 2. Build a candidate set S from the path and its neighbors.
    candidate_set = set(base_path)
    for poi_name in base_path:
        poi = poi_map.get(poi_name)
        if poi and 'neighbors' in poi and isinstance(poi['neighbors'], dict):
            candidate_set.update(poi['neighbors'].keys())

            # If no path, use all points as candidates
    if not candidate_set:
        candidate_set = set(poi_map.keys())

        # Filter out start, end, and already selected waypoints
    candidate_names = [name for name in candidate_set if name not in [start_pt, end_pt] + st.session_state.waypoints]

    if not candidate_names:
        return []

        # 3. Calculate similarity for each candidate using its 'feature' list.
    scores = {}
    for name in candidate_names:
        poi = poi_map.get(name)
        if poi and 'feature' in poi and poi['feature']:
            poi_concept = " ".join(poi['feature'])
            relevance_score = similarity_calculator.calculate(user_demand, poi_concept)
            scores[name] = relevance_score

    if not scores:
        return []

        # 4. Return the top 3 most similar locations.
    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [name for name, score in sorted_candidates[:3]]


def handle_route_change():
    """Callback to clean up waypoints and suggestions if the start/end point changes."""
    start = st.session_state.start_point
    end = st.session_state.end_point
    st.session_state.waypoints = [wp for wp in st.session_state.waypoints if wp not in [start, end]]
    st.session_state.suggestions = []  # Clear old suggestions


def generate_map_html(map_data, start_pt, end_pt, way_pts, calc_path=None):
    """Generates the Folium map HTML based on the current state."""
    m = folium.Map(location=[1.3521, 103.8198], zoom_start=11)

    marker_cluster = MarkerCluster(name="All Points of Interest").add_to(m)
    for name, poi in map_data.items():
        popup = f"<b>{name}</b><hr>Category: {poi.get('category', 'N/A')}"
        if poi.get('photo_url'):
            popup += f"<img src='{poi['photo_url']}' width='200'<br><br>"
        popup += f"<b>Category:</b> {poi.get('category', 'N/A')}<br>"
        if poi.get('feature'):
            popup += f"<b>Features:</b> {', '.join(poi.get('feature', []))}<br>"
        popup += f"<b>Rating:</b> {poi.get('rating', 'N/A')}<br>"
        popup += f"<b>Popularity:</b> {poi.get('review_count', 'N/A')}<br>"
        popup += f"<b>Recommend Score:</b> {poi.get('reward', 'N/A')}<br>"
        folium.Marker(location=[poi['lat'], poi['lng']], popup=popup, tooltip=name).add_to(marker_cluster)

    selection_fg = folium.FeatureGroup(name="Your Selections", show=True).add_to(m)
    selections = [(start_pt, "start"), (end_pt, "end")] + [(wp, "waypoint") for wp in way_pts]
    for pt, type in selections:
        if pt and pt in map_data:
            poi = map_data[pt]
            icon_color = {'start': 'green', 'end': 'red', 'waypoint': 'orange'}[type]
            icon_shape = {'start': 'play', 'end': 'stop', 'waypoint': 'flag'}[type]
            selection_fg.add_child(
                folium.Marker(location=[poi['lat'], poi['lng']],
                              popup=f"<b>{type.title()}:</b> {pt}",
                              icon=folium.Icon(color=icon_color, icon=icon_shape, prefix='fa')))

    if calc_path:
        route_fg = folium.FeatureGroup(name="Calculated Route", show=True).add_to(m)
        path_latlngs = [(map_data[name]['lat'], map_data[name]['lng']) for name in calc_path if name in map_data]

        # MODIFICATION START: Add markers for intermediate points on the path
        for i, name in enumerate(calc_path):
            # Highlight intermediate points that are not the start, end, or a user-set waypoint
            if name not in [start_pt, end_pt] and name not in way_pts:
                if name in map_data:
                    poi = map_data[name]
                    folium.Marker(
                        location=[poi['lat'], poi['lng']],
                        popup=f"<b>Step {i + 1}:</b> {name}",
                        tooltip=f"Step {i + 1}: {name}",
                        icon=folium.Icon(color='blue', icon='map-pin', prefix='fa')
                    ).add_to(route_fg)
                    # MODIFICATION END

        if path_latlngs:
            folium.PolyLine(path_latlngs, color="#FF0000", weight=5, opacity=0.8).add_to(route_fg)
            m.fit_bounds(route_fg.get_bounds(), padding=(30, 30))

    folium.LayerControl().add_to(m)

    folium.LayerControl().add_to(m)
    return m.get_root().render()


# --- 4. Streamlit App Execution ---
st.set_page_config(layout="wide")
st.title("GUIDE - Garden city's Unique and Intelligent Discovery Engine")

# Load base data/tools and set up session state on first run
pristine_graph_data = load_pristine_data(BASE_GRAPH_FILE)
similarity_calculator = get_similarity_calculator()
initialize_session_state()

all_names = sorted([poi['name'] for poi in pristine_graph_data])
poi_data_map = {poi['name']: poi for poi in pristine_graph_data}

# --- Sidebar UI ---
st.sidebar.header("Plan Your Personalized Trip")
user_demand = st.sidebar.text_area("1. Tell us what you're looking for to get suggestions", key='user_demand_input')
st.sidebar.markdown("---")
st.sidebar.markdown("##### 2. Select Your Route")

start_default_idx = all_names.index("Merlion Park") if "Merlion Park" in all_names else 0
end_default_idx = all_names.index(
    "National University of Singapore") if "National University of Singapore" in all_names else 1

start_point = st.sidebar.selectbox("Start", all_names, index=start_default_idx, key='start_point',
                                   on_change=handle_route_change)
end_point = st.sidebar.selectbox("End", all_names, index=end_default_idx, key='end_point',
                                 on_change=handle_route_change)
waypoints = st.sidebar.multiselect("Passing Points", [n for n in all_names if n not in [start_point, end_point]],
                                   key='waypoints', default=st.session_state.waypoints)

# --- UI for Waypoint Suggestion and Path Calculation ---
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Suggest Waypoints", use_container_width=True):
        with st.spinner("Finding the best matches..."):
            suggestions = find_best_waypoint(pristine_graph_data, start_point, end_point, user_demand)
            st.session_state.suggestions = suggestions
            if not suggestions:
                st.sidebar.warning("No suitable suggestions found.")
with col2:
    calculate_button = st.button("Let's Go!", type="primary", use_container_width=True)

# --- Display Interactive Suggestions ---
if st.session_state.get('suggestions'):
    st.sidebar.markdown("---")
    st.sidebar.markdown("✨ **Top Suggestions**")
    st.sidebar.caption("Click to add to your passing points.")
    for suggestion in st.session_state.suggestions:
        if st.sidebar.button(f"⭐ {suggestion}", key=f"sugg_{suggestion}", use_container_width=True):
            if suggestion not in st.session_state.waypoints:
                st.session_state.waypoints.append(suggestion)
                st.toast(f"Added '{suggestion}' to your waypoints!", icon="✨")
                st.rerun()  # Rerun to update the multiselect widget
            else:
                st.toast(f"'{suggestion}' is already in your list.", icon="👍")

            # --- Main Logic for Path Calculation ---
if calculate_button:
    with st.spinner("Calculating the optimal path..."):
        # Pathfinding now uses the base file, as there's no personalized session file.
        path_result = find_path(BASE_GRAPH_FILE, start_point, waypoints, end_point, "distance")
        st.session_state.calc_path = path_result.get('path') if path_result else None

        if st.session_state.calc_path:
            st.sidebar.success(f"Path found! Visiting {len(st.session_state.calc_path)} locations.")
            with st.sidebar.expander("Your Itinerary", expanded=True):
                for i, loc_name in enumerate(st.session_state.calc_path):
                    if loc_name == start_point:
                        label = "🟢 Start"
                    elif loc_name == end_point:
                        label = "🔴 End"
                    else:
                        label = f"{i + 1}"
                    st.write(f"**{label}.** {loc_name}")
        else:
            st.sidebar.warning("Could not find a path. Try another combination.")

    # --- Display Map ---
map_html = generate_map_html(poi_data_map, start_point, end_point, st.session_state.waypoints,
                             st.session_state.calc_path)
st.components.v1.html(map_html, height=650)
