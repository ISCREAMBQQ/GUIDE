import streamlit as st
import json
import folium
from folium.plugins import MarkerCluster

# Make sure your path algorithm files are correctly placed
from path_algorithm import find_path


# --- Performance Optimization: Use Streamlit's cache to load and preprocess data ---
@st.cache_data
def load_poi_data(file_path):
    """
    Loads and processes POI data from a JSON file.
    The result of this function will be cached, running only on the first load or when the file changes.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Create a dictionary for quick lookups and filter out any invalid data without coordinates
        poi_dict = {
            item['name']: item
            for item in data if isinstance(item.get('lat'), (int, float)) and isinstance(item.get('lng'), (int, float))
        }
        return poi_dict
    except FileNotFoundError:
        return {"error": "File not found"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}


# --- Streamlit App Body ---

# 1. Page Configuration
st.set_page_config(layout="wide")
st.title("GUIDE - Garden city's Unique and Intelligent Discovery Engine")

# 2. Load Data (leveraging the cache)
poi_data = load_poi_data("Graph/GUIDE_037_updated.json")

# If data loading fails, show an error and stop the app
if "error" in poi_data:
    st.error(f"Data loading failed: {poi_data['error']}. Please check the file path and content.")
    st.stop()

poi_dict = poi_data
all_names = sorted(list(poi_dict.keys()))  # Sort names for a better user experience

# 3. Sidebar: User Input Controls
st.sidebar.header("Where do you want to go?")
st.sidebar.info("💡 You can type keywords below to search for your destination.")

# Set sensible defaults to make the app more engaging on first load
start_default = all_names.index("Merlion Park") if "Merlion Park" in all_names else 0
end_default = all_names.index(
    "National University of Singapore") if "National University of Singapore" in all_names else 1

# --- FIX APPLIED HERE: Added unique 'key' to each widget ---
# The key gives the widget a "memory" across script re-runs.
start_point = st.sidebar.selectbox(
    "Your Location (Start)",
    all_names,
    index=start_default,
    key='start_point_selector'  # Added key
)
end_point = st.sidebar.selectbox(
    "Your Destination (End)",
    all_names,
    index=end_default,
    key='end_point_selector'  # Added key
)

# The list of available waypoints should exclude the chosen start and end points
available_waypoints = [n for n in all_names if n not in [start_point, end_point]]
waypoints = st.sidebar.multiselect(
    "Passing Points (Optional)",
    available_waypoints,
    key='waypoints_selector'  # Added key
)

calculate_button = st.sidebar.button("Let's Go!", type="primary", use_container_width=True)

# 4. Map Creation and Path Calculation
# Initialize the map centered on Singapore
m = folium.Map(location=[1.3521, 103.8198], zoom_start=11)

# --- Performance Optimization: Use MarkerCluster for all POI markers ---
marker_cluster = MarkerCluster(name="All POIs").add_to(m)
for name, poi in poi_dict.items():
    popup_html = f"""
    <b>{name}</b><hr>
    <b>Category:</b> {poi.get('category', 'N/A')}<br>
    <b>Rating:</b> {poi.get('rating', 'N/A')} ({poi.get('review_count', 0)} reviews)<br>
    <b>Reward Score:</b> {poi.get('reward', 0):.2f}
    """
    folium.Marker(
        location=[poi['lat'], poi['lng']],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=name  # Show place name on hover
    ).add_to(marker_cluster)

# Execute path calculation and drawing only when the button is clicked
if calculate_button:
    with st.spinner("Calculating the optimal path, please wait..."):
        try:
            json_file = "Graph/GUIDE_037_updated.json"
            # Call your pathfinding function which should handle start, waypoints, and end
            path_result = find_path(json_file, start_point, waypoints, end_point, "distance")
            calc_path = path_result.get('path') if path_result else None

            if not calc_path:
                st.sidebar.warning("Could not find a path. This route might be very unpopular, try another one?")
            else:
                st.sidebar.success(f"Path found! You will visit {len(calc_path)} locations.")
                path_latlngs = [
                    (poi_dict[name]['lat'], poi_dict[name]['lng'])
                    for name in calc_path if name in poi_dict
                ]

                # Draw the calculated path on the map
                if path_latlngs:
                    folium.PolyLine(
                        path_latlngs, color="#FF0000", weight=5, opacity=0.8, tooltip="Calculated Route"
                    ).add_to(m)

                    # Add special markers for start, end, and waypoints
                    # Start Point
                    folium.Marker(
                        location=path_latlngs[0], popup=f"<b>Start:</b><br>{start_point}",
                        icon=folium.Icon(color='green', icon='play', prefix='fa')
                    ).add_to(m)
                    # End Point
                    folium.Marker(
                        location=path_latlngs[-1], popup=f"<b>End:</b><br>{end_point}",
                        icon=folium.Icon(color='red', icon='stop', prefix='fa')
                    ).add_to(m)
                    # Waypoints
                    for waypoint_name in waypoints:
                        if waypoint_name in poi_dict:
                            folium.Marker(
                                location=[poi_dict[waypoint_name]['lat'], poi_dict[waypoint_name]['lng']],
                                popup=f"<b>Waypoint:</b><br>{waypoint_name}",
                                icon=folium.Icon(color='orange', icon='flag', prefix='fa')
                            ).add_to(m)

                    # Automatically zoom the map to fit the path
                    m.fit_bounds(folium.PolyLine(path_latlngs).get_bounds(), padding=(30, 30))

                # --- Display the Itinerary in the Sidebar ---
                with st.sidebar.expander("Your Itinerary", expanded=True):
                    for i, location_name in enumerate(calc_path):
                        if i == 0:
                            st.write(f"**{i + 1}.** {location_name} (🟢 Start)")
                        elif i == len(calc_path) - 1:
                            st.write(f"**{i + 1}.** {location_name} (🔴 End)")
                        else:
                            st.write(f"**{i + 1}.** {location_name}")

        except Exception as e:
            st.sidebar.error(f"An error occurred during path calculation: {e}")

# 5. Render the final map in Streamlit
# This renders the map with all the updates made above (clusters, path, special markers)
st.components.v1.html(m.get_root().render(), height=600)

st.info("The map is interactive. Zoom in and out, or click on the clustered markers to explore more locations.")
