### JUPYTER MARKDOWN CELL ###
# Problem Motivation
# On average, about one 10m asteroid passes within lunar distance of the Earth each day. Some are too small to be detected before they pass, and a still larger portion are suspected to pass by whilst escaping detection altogether. Most people are unaware of these cosmic encounters, and those that do make the news are largely sensationalized, leading to public panic, or overly technical, causing a lack of accessibility. Thus, we wanted to create a system capable of monitoring real data on the positions and trajectories of asteroids, contextualizing risk in a readable manner, and distinguishing between routine passes and dangerous anomalies. 
# This project addresses these challenges by:
# - Automatically fetching and processing current asteroid approach data
# - Organizing temporal patterns to identify trends
# - Applying scientific criteria to assess actual risk levels
# - Generating clear, contextual explanations suitable for public consumption

### JUPYTER MARKDOWN CELL ###
# Setup and Dependencies
# This cell begins importing the necessary packages and checks that Skyfield is installed. The `skyfield` library is used for accurate planetary position calculations in our 3D solar system visualization.

### JUPYTER MARKDOWN CELL ###
# Import Libraries
# This cell imports all the necessary libraries for data fetching, processing, and visualization.

import requests
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import json
from datetime import datetime, timedelta
import threading
import tkinter as tk
from tkinter import ttk
import time
# Removed: from matplotlib.patches import Rectangle

### JUPYTER MARKDOWN CELL ###
# Configuration and Constants
# This cell defines API endpoints, file paths, and scientific thresholds used throughout the analysis.

# --- 0. SCRIPT CONFIGURATION ---

# Output settings
OUTPUT_DIR = Path("plots")
RETENTION_DAYS = 3  # files older than this will be deleted
CACHE_DIR = Path("cache")
SBDB_CACHE_DIR = CACHE_DIR / "sbdb"
SBDB_CACHE_TTL_DAYS = 30

# Define the API endpoint
API_URL = "https://ssd-api.jpl.nasa.gov/cad.api"
SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"

# --- CONSTANTS ---
AU_TO_KM = 149597870.7
AU_TO_LD = 389.5
PHA_H_THRESHOLD = 22.0
# Calculate 10 Lunar Distances in AU
NON_HAZ_CLOSE_AU = 10.0 / AU_TO_LD

### JUPYTER MARKDOWN CELL ###
# Helper Functions
# This cell includes utility functions for API communication, caching, file management, and coordinate transformations. Functions here create and manage the loading window that displays progress during data fetching. The loading window runs on the main thread while data fetching happens in a background thread. This prevents the GUI from freezing during the 10-30 second API query process.

# --- 1. HELPER FUNCTIONS (GUI, FILE, AND NETWORK) ---

### JUPYTER MARKDOWN CELL ###
# Progress Update and Window Management Functions
# These utility functions handle critical elements for a good user experience, including keeping the loading window responsive during data fetching and ensuring plots display properly across different systems. The `update_progress()` function enables thread-safe GUI updates. Since the data pipeline runs in a background thread (to prevent UI freezing during the API fetch), we need a safe way to update the progress bar and status text from that thread. This function checks that widgets still exist before updating them, preventing crashes if the user closes the window early. Matplotlib can use different GUI backends (TkAgg, Qt, wxPython, etc.) depending on the user's system, and each has different methods for maximizing windows. This function attempts all common approaches in sequence, ensuring our large visualization opens maximized for optimal viewing, regardless of backend. If all methods fail, the window simply opens at default size—the function fails gracefully without breaking the program.

def create_loading_window():
    """Creates and returns a simple tkinter loading window."""
    root = tk.Tk()
    root.title("Loading Asteroid Data")
    
    # Center the window
    window_width = 350
    window_height = 100
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_cordinate = int((screen_width/2) - (window_width/2))
    y_cordinate = int((screen_height/2) - (window_height/2))
    root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")
    
    root.resizable(False, False)
    root.attributes("-topmost", True)
    
    frame = ttk.Frame(root, padding="20")
    frame.pack(expand=True, fill="both")
    
    status_label = ttk.Label(frame, text="Initializing...", font=("Helvetica", 11))
    status_label.pack(pady=5)
    
    progress = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate")
    progress.pack(pady=5)
    
    return root, status_label, progress

def update_progress(label, progress_bar, text, value):
    """Thread-safe way to update the loading window's status and progress bar."""
    if label and label.winfo_exists():
        label.config(text=text)
    if progress_bar and progress_bar.winfo_exists():
        progress_bar['value'] = value

def maximize_figure_window(fig):
    """Attempts to maximize the matplotlib plot window."""
    try:
        manager = plt.get_current_fig_manager()
        # TkAgg on Windows
        if hasattr(manager, 'window') and hasattr(manager.window, 'state'):
            try:
                manager.window.state('zoomed')
                return
            except Exception:
                pass
        # Qt backend
        if hasattr(manager, 'window') and hasattr(manager.window, 'showMaximized'):
            try:
                manager.window.showMaximized()
                return
            except Exception:
                pass
        # wx backend
        if hasattr(manager, 'frame') and hasattr(manager.frame, 'Maximize'):
            try:
                manager.frame.Maximize(True)
                return
            except Exception:
                pass
    except Exception:
        pass

### JUPYTER MARKDOWN CELL ###
# Network, File Management, and Caching Functions
# These functions handle robust API communication and efficient data management. The `create_session_with_retries()` function creates an HTTP session that automatically retries failed requests. This is critical for the NASA API, which can experience intermittent network issues. The file management functions prevent disk bloat and caches orbital data by removing old visualizations to avoid redundant API calls. Orbital elements rarely change, so caching them every 30 days dramatically reduces API load when re-running analyses.

def create_session_with_retries(total_retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    """Creates a requests.Session with automatic retries."""
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def cleanup_old_files(directory: Path, days: int) -> None:
    """Removes old .png files from the specified directory."""
    if not directory.exists():
        return
    cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=days)
    for p in directory.glob("*.png"):
        try:
            mtime = pd.Timestamp(p.stat().st_mtime, unit='s')
            if mtime < cutoff:
                p.unlink(missing_ok=True)
        except Exception:
            pass

def load_json_cache(cache_path: Path, max_age_days: int):
    try:
        if not cache_path.exists():
            return None
        age_days = (pd.Timestamp.now(tz=None) - pd.Timestamp(cache_path.stat().st_mtime, unit='s')).days
        if age_days > max_age_days:
            return None
        with cache_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def save_json_cache(cache_path: Path, data: dict):
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass

# --- 2. DATA PIPELINE (This function runs in a separate thread) ---

### JUPYTER MARKDOWN CELL ###
# Data Pipeline - Fetching and Processing Asteroid Data
# This cell is the core data processing function that runs in a background thread while the loading window displays progress. It performs four main stages: 
# 
# 1. fetching 6-month windows of asteroid data from NASA's API in batches
# 2. cleaning and parsing the raw JSON into structured DataFrames
# 3. calculating derived metrics and applying PHA classification criteria
# 4. filtering into time-window subsets for visualization. 
# 
# The function updates the progress bar at each stage and stores results in global variables accessible to the plotting functions. The API has URL length limits and performs better with smaller date ranges. We split the 6-month window into six 15-30 day chunks, fetched them sequentially, then concatenated them. This also enables granular progress reporting so users know the system hasn't frozen.

def run_data_pipeline(status_label, progress_bar, on_complete_callback):
    """
    Main function to fetch and process data.
    Designed to run in a worker thread.
    """
    global pipeline_results  # Store results in a global variable
    
    try:
        # --- PART A: DATA FETCHING ---
        
        api_configs = [
            {'date-min': 'now', 'date-max': '+15', 'dist-max': '0.5'},
            {'date-min': '+16', 'date-max': '+30', 'dist-max': '0.5'},
            {'date-min': '+31', 'date-max': '+45', 'dist-max': '0.5'},
            {'date-min': '+46', 'date-max': '+60', 'dist-max': '0.5'},
            {'date-min': '+61', 'date-max': '+120', 'dist-max': '0.5'},
            {'date-min': '+121', 'date-max': '+180', 'dist-max': '0.5'}
        ]
        
        all_dataframes = []
        session = create_session_with_retries()
        
        for i, config in enumerate(api_configs):
            status_text = f"Fetching data ({i+1}/6): {config['date-min']} to {config['date-max']}..."
            progress_value = 10 + (i * 10) 
            update_progress(status_label, progress_bar, status_text, progress_value)

            params = {
                **config,
                'sort': 'dist',
                'limit': '10000',
                'diameter': 'true'
            }
            
            response = session.get(API_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data and data['data']:
                fields = data['fields']
                all_dataframes.append(pd.DataFrame(data['data'], columns=fields))

        if not all_dataframes:
            update_progress(status_label, progress_bar, "No data returned from API. Exiting.", 100)
            on_complete_callback()
            return
            
        df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Successfully fetched {len(df)} total close approaches.")

        # --- PART B: DATA PARSING & CLEANING ---
        update_progress(status_label, progress_bar, "Cleaning and parsing data...", 70)

        cols_to_convert = ['dist', 'dist_min', 'dist_max', 'v_rel', 'h', 'diameter']
        for col in cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['cd_datetime'] = pd.to_datetime(df['cd'])
        df = df.dropna(subset=['dist', 'h', 'cd_datetime'])

        # --- PART C: DATA ANALYSIS & "THE MATH" ---
        update_progress(status_label, progress_bar, "Analyzing and filtering...", 90)

        df['dist_km'] = df['dist'] * AU_TO_KM
        df['dist_lunar'] = df['dist'] * AU_TO_LD # Still useful for sorting
        df['is_hazardous'] = ((df['dist'] <= 0.05) & (df['h'] <= PHA_H_THRESHOLD))

        now = datetime.now()
        plus_7_days = now + timedelta(days=7) # Used for df_haz_30_days start point
        plus_30_days = now + timedelta(days=30)
        plus_60_days = now + timedelta(days=60)
        plus_6_months = now + timedelta(days=180)

        df_non_hazardous_close = df[
            (df['cd_datetime'] <= plus_60_days) & 
            (df['dist'] <= NON_HAZ_CLOSE_AU) &
            (df['is_hazardous'] == False)
        ].copy()

        df_haz_30_days = df[
            (df['cd_datetime'] > plus_7_days) & 
            (df['cd_datetime'] <= plus_30_days) & 
            (df['is_hazardous'] == True)
        ].sort_values(by='dist_lunar').head(25)

        df_haz_6_months = df[
            (df['cd_datetime'] > plus_30_days) & 
            (df['cd_datetime'] <= plus_6_months) & 
            (df['is_hazardous'] == True)
        ].sort_values(by='dist_lunar').head(25)

        # --- PART D: STORE RESULTS ---
        pipeline_results = (df_non_hazardous_close, df_haz_30_days, df_haz_6_months)
        update_progress(status_label, progress_bar, "Data processed. Finishing up...", 100)
        
    except Exception as e:
        print(f"An error occurred in the data pipeline: {e}")
        update_progress(status_label, progress_bar, f"Error: {e}", 100)
    finally:
        time.sleep(0.5) # So user can see 100%
        on_complete_callback()

# --- 3A. ORBIT/ASTRONOMY UTILITIES ---

### JUPYTER MARKDOWN CELL ###
# Astronomical Calculation Functions
# These functions perform the mathematical transformations needed to visualize asteroid and planet orbits in 3D space. They convert Keplerian orbital elements into Cartesian coordinates (x, y, z positions in space) so they can be plotted. This is complex because orbits are naturally described using 6 parameters (semi-major axis, eccentricity, inclination, etc.), but to draw them on a 3D plot, we need Cartesian coordinates. This requires solving Kepler's equation, a transcendental equation with no closed-form solution, and applying 3D rotation matrices to transform from the orbital plane to the ecliptic reference frame. These functions implement the classical orbital mechanics algorithms used throughout astronomy and spaceflight. In this case, AI was particularly useful in producing the necessary matrices and equations to code this process.

def kepler_solve_eccentric_anomaly(mean_anomaly_rad, eccentricity, tol=1e-10, max_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) for E using Newton-Raphson."""
    M = np.mod(mean_anomaly_rad, 2 * np.pi)
    if eccentricity < 0.8:
        E = M
    else:
        E = np.pi
    for _ in range(max_iter):
        f = E - eccentricity * np.sin(E) - M
        f_prime = 1 - eccentricity * np.cos(E)
        delta = -f / f_prime
        E = E + delta
        if np.all(np.abs(delta) < tol):
            break
    return E

def orbital_elements_to_position_au(a_au, e, i_deg, omega_deg, w_deg, M_deg):
    """
    Convert Keplerian elements to heliocentric ecliptic position (AU).
    Angles in degrees. Returns (x, y, z) in AU.
    """
    i = np.deg2rad(i_deg)
    Omega = np.deg2rad(omega_deg)
    w = np.deg2rad(w_deg)
    M = np.deg2rad(M_deg)

    E = kepler_solve_eccentric_anomaly(M, e)
    # True anomaly
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a_au * (1 - e * np.cos(E))

    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0.0

    # Rotation: argument of periapsis, inclination, longitude of ascending node
    cO, sO = np.cos(Omega), np.sin(Omega)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(w), np.sin(w)

    # Rotation matrix R = Rz(Omega) * Rx(i) * Rz(w)
    R11 = cO * cw - sO * sw * ci
    R12 = -cO * sw - sO * cw * ci
    R13 = sO * si
    R21 = sO * cw + cO * sw * ci
    R22 = -sO * sw + cO * cw * ci
    R23 = -cO * si
    R31 = sw * si
    R32 = cw * si
    R33 = ci

    x = R11 * x_orb + R12 * y_orb + R13 * z_orb
    y = R21 * x_orb + R22 * y_orb + R23 * z_orb
    z = R31 * x_orb + R32 * y_orb + R33 * z_orb
    return float(x), float(y), float(z)

### JUPYTER MARKDOWN CELL ###
# Planet and Asteroid Orbital Data
# These functions retrieve orbital elements from NASA's Small-Body Database and provide fixed, immutable data for the inner planets. The `try_fetch_sbdb_elements()` function queries the SBDB API for a specific asteroid's orbital parameters and caches them locally to avoid redundant requests, since orbital elements change slowly. The `build_planet_catalog()` provides accurate orbital data for Mercury through Mars used in the 3D solar system visualization.

def try_fetch_sbdb_elements(designation):
    """Fetch orbital elements from SBDB; returns dict with a,e,i,om,w,ma (deg)."""
    try:
        # Try cache first
        safe_des = str(designation).replace('/', '_').replace(' ', '_')
        cache_file = SBDB_CACHE_DIR / f"{safe_des}.json"
        cached = load_json_cache(cache_file, SBDB_CACHE_TTL_DAYS)
        if cached and isinstance(cached, dict):
            # Check if cache is the valid, new format
            if "a" in cached and "e" in cached:
                return cached
            # If old format, just fall through to re-fetch
            
        resp = requests.get(SBDB_URL, params={"des": designation}, timeout=15)
        resp.raise_for_status()
        j = resp.json()

        # Handle cases where API returns a list (ambiguous query)
        if isinstance(j, list):
            print(f"ERROR: Ambiguous designation {designation}. API returned list.")
            return None
        
        orbit_data = j.get("orbit") or {} 
        
        if isinstance(orbit_data, list):
            if not orbit_data:
                print(f"ERROR: API returned an empty orbit list for {designation}")
                return None
            orbit = orbit_data[0] # Take the first orbit from the list
        else:
            orbit = orbit_data # It was a dictionary as expected
        
        # --- START OF FIX (Changelog 4) ---
        # The 'elements' key contains a LIST of dictionaries, not a single dictionary.
        # We must parse this list.
        elements_list = orbit.get("elements")
        if not elements_list or not isinstance(elements_list, list):
            print(f"ERROR: No 'elements' list found in orbit object for {designation}")
            return None
        
        # Convert the list of elements into a dictionary
        elements_dict = {}
        for item in elements_list:
            if isinstance(item, dict) and 'name' in item and 'value' in item:
                elements_dict[item['name']] = item['value']
        
        # Now extract values from the new dictionary, with safety checks
        def get_float(name):
            """Safely get a float from the parsed elements dictionary."""
            try:
                # Use .get() which returns None if key doesn't exist
                return float(elements_dict.get(name))
            except (TypeError, ValueError, TypeError):
                # This catches if value is None or not a valid float string
                return None

        a = get_float('a')
        e = get_float('e')
        i = get_float('i')
        om = get_float('om')
        w = get_float('w')
        ma = get_float('ma')
        n = get_float('n')
        epoch = get_float('epoch')

        # Check if essential elements were found (n and epoch are optional)
        if any(v is None for v in [a, e, i, om, w, ma]):
             print(f"ERROR: Missing one or more essential orbit elements (a, e, i, om, w, ma) for {designation}")
             print(f"Found elements: {elements_dict}") # Show what we *did* find
             return None

        result = {"a": a, "e": e, "i": i, "om": om, "w": w, "ma": ma, "n": n, "epoch": epoch}
        # --- END OF FIX ---

        save_json_cache(cache_file, result)
        return result
    except Exception as e:
        print(f"ERROR in try_fetch_sbdb_elements for {designation}: {e}")
        return None

def build_planet_catalog():
    """Return simple planet data up to Mars with semi-major axis and radii (km)."""
    # Basic values; sizes will be log-scaled when displayed
    return [
        {"name": "Mercury", "a": 0.387098, "e": 0.205630, "i": 7.0049, "om": 48.331, "w": 29.124, "ma": 174.796, "radius_km": 2439.7},
        {"name": "Venus",   "a": 0.723332, "e": 0.006772, "i": 3.3947, "om": 76.680, "w": 54.884, "ma": 50.115,  "radius_km": 6051.8},
        {"name": "Earth",   "a": 1.000000, "e": 0.016710, "i": 0.0000, "om": -11.260, "w": 114.207, "ma": 358.617, "radius_km": 6371.0},
        {"name": "Mars",    "a": 1.523679, "e": 0.093400, "i": 1.850,  "om": 49.558, "w": 286.502, "ma": 19.373,  "radius_km": 3389.5},
    ]

def log_scale_sizes(radii_km, min_size=20, max_size=80):
    r = np.array(radii_km, dtype=float)
    r = np.maximum(r, 1.0)
    logr = np.log10(r)
    logr = (logr - logr.min()) / (logr.max() - logr.min() + 1e-9)
    return min_size + logr * (max_size - min_size)

def julian_date_utc(now_dt=None):
    now = now_dt or datetime.utcnow()
    year = now.year
    month = now.month
    day = now.day + (now.hour + (now.minute + now.second/60.0)/60.0)/24.0
    if month <= 2:
        year -= 1
        month += 12
    A = int(year/100)
    B = 2 - A + int(A/4)
    jd = int(365.25*(year + 4716)) + int(30.6001*(month + 1)) + day + B - 1524.5
    return jd

### JUPYTER MARKDOWN CELL ###
# Skyfield Integration & 3D Visualization
# This cell implements the interactive 3D solar system visualization and precise planet position calculations. The `init_skyfield_if_available()` function dynamically imports the Skyfield library and loads the DE421 ephemeris for accurate planetary positions. The `get_planet_positions_au_with_skyfield()` function queries real-time heliocentric positions in ecliptic coordinates, providing significantly more accurate planet locations than simplified Keplerian elements.
# The `create_and_show_plot()` function orchestrates the entire visualization dashboard using a complex matplotlib GridSpec layout: a left text panel for documentation and four right panels for scatter plots and the 3D solar system map. The 3D view includes planet orbits represented by yellow curves, current positions with directional arrows, and a subtle asteroid belt band. An interactive click handler allows users to select any asteroid point from the other graphs, dynamically overlaying its full orbital trajectory in red with directional arrows on the 3D map. This provides an intuitive way to explore how specific asteroids relate spatially to the inner solar system.

_skyfield_cache = {"loaded": False, "eph": None, "ts": None, "ecliptic": None}

def init_skyfield_if_available():
    if _skyfield_cache["loaded"]:
        return True
    # Local dynamic import to avoid static unresolved-import warnings
    import importlib
    _api = importlib.import_module('skyfield.api')
    _fr = importlib.import_module('skyfield.framelib')
    load = _api.load
    _ecliptic_frame = _fr.ecliptic_frame
    ts = load.timescale()
    eph = load("de421.bsp")
    _skyfield_cache.update({"loaded": True, "eph": eph, "ts": ts, "ecliptic": _ecliptic_frame})
    return True

def get_planet_positions_au_with_skyfield():
    """Return dict of planet name -> (x,y,z) AU in ecliptic frame, if available."""
    if not init_skyfield_if_available():
        return None
    eph = _skyfield_cache["eph"]
    ts = _skyfield_cache["ts"]
    t = ts.now()
    sun = eph['sun']
    bodies = {
        "Mercury": eph['mercury barycenter'],
        "Venus": eph['venus barycenter'],
        "Earth": eph['earth barycenter'],
        "Mars": eph['mars barycenter'],
    }
    result = {}
    for name, body in bodies.items():
        vec = (body - sun).at(t).frame_latlon(_skyfield_cache["ecliptic"])
        # Convert from spherical ecliptic to Cartesian in AU using distance + angles
        # frame_latlon returns (lat, lon, distance), so use .distance().au and angles in radians
        lat, lon, distance = vec
        r = distance.au
        latr = lat.radians
        lonr = lon.radians
        x = r * np.cos(latr) * np.cos(lonr)
        y = r * np.cos(latr) * np.sin(lonr)
        z = r * np.sin(latr)
        result[name] = (x, y, z)
    return result

# --- 3B. PLOTTING FUNCTION (This runs on the main thread) ---

def create_and_show_plot(df_non_hazardous_close, df_haz_30_days, df_haz_6_months):
    """
    Creates, saves, and shows the matplotlib plot.
    This function MUST run on the main thread.
    """
    print("Generating plot...")
    
    # Use GridSpec for a complex layout (text panel + 2x2 grid)
    fig = plt.figure(figsize=(19.2, 12.8))
    fig.patch.set_facecolor('black')
    
    # Create a 2-row, 3-column grid: left column is text (narrower), two right columns are equal-size plots
    gs = fig.add_gridspec(2, 3, width_ratios=[0.6, 1.0, 1.0])

    # Create the text axis on the left (spanning both rows)
    ax_text = fig.add_subplot(gs[:, 0]) # gs[:, 0] means "all rows, column 0"
    
    # Create the 4 plot axes on the right (all equal sizes now)
    axes_list = [
        fig.add_subplot(gs[0, 1]),        # Graph 1: top-left
        fig.add_subplot(gs[0, 2]),        # Graph 2: top-right
        fig.add_subplot(gs[1, 1]),        # Graph 3: bottom-left
        fig.add_subplot(gs[1, 2], projection='3d')  # Graph 4: bottom-right (3D Solar System)
    ]
    
    # --- Add descriptive text to the left panel ---
    ax_text.set_facecolor('black')
    ax_text.axis('off') # Hide the axis borders, ticks, and labels

    # Define the text content
    info_text = (
        "About This Data\n\n"
        "This dashboard visualizes near-Earth\n"
        "asteroid close approaches.\n\n\n"
        "Data Source:\n"
        "NASA/JPL SBDB Close Approach API.\n\n\n"
        "What is 'Hazardous'?\n"
        "Potentially Hazardous (PHA) objects meet:\n"
        "  •  H ≤ 22.0 (~>140 m)\n"
        "  •  Distance ≤ 0.05 AU (~19.5 LD)\n\n\n"
        "3D Solar System Map:\n"
        "Sun at origin; Mercury–Mars orbits\n"
        "shown in yellow with arrows.\n"
        "Planet sizes use log scaling.\n"
        "Click an asteroid point to overlay\n"
        "its orbit in red."
    )

    # --- CHANGE 1: Removed the beveled box patches ---

    # --- CHANGE 2: Add the text, set to white, reduce font size ---
    ax_text.text(0.05, 0.985, info_text,
                 transform=ax_text.transAxes,
                 fontsize=9,      # Reduced from 10 to 9
                 color='white',   # Changed from black
                 va='top',
                 ha='left',
                 wrap=True)
                 # Removed zorder

    # Add the main title
    fig.suptitle('Asteroid Close Approaches - Proximity & Hazard Over Time', fontsize=20, color='white', y=0.98)

    # Re-ordered: move former Graph 3 to 2, former Graph 4 to 3; Graph 4 is new 3D map
    plot_data_frames = [
        (df_non_hazardous_close, "Non-Hazardous, Relatively Close (Next 60 Days)", {'is_hazardous': False}),
        (df_haz_30_days, "Top 25 Hazardous (8-30 Days)", {'is_hazardous': True}),
        (df_haz_6_months, "Top 25 Hazardous (31-180 Days)", {'is_hazardous': True})
    ]

    max_dist_30 = df_haz_30_days['dist'].max() if not df_haz_30_days.empty else 0
    max_dist_180 = df_haz_6_months['dist'].max() if not df_haz_6_months.empty else 0
    uniform_xlim_max = max(max_dist_30, max_dist_180, 0.05)

    # Storage for click mapping
    point_metadata = []  # list of (ax, PathCollection, df)

    for i, (current_df, title, filter_condition) in enumerate(plot_data_frames):
        ax = axes_list[i]
        
        x_col = 'dist'
        y_col = 'h'
        
        if filter_condition.get('is_hazardous'):
            colors = 'red'
            label_text = 'Potentially Hazardous'
        else:
            colors = 'gray'
            label_text = 'Non-Hazardous'

        if i == 0:
            if not current_df.empty:
                xlim_max = current_df[x_col].max()
            else:
                xlim_max = NON_HAZ_CLOSE_AU
            
            min_xlim_au = 12.0 / AU_TO_LD
            if xlim_max < min_xlim_au:
                xlim_max = min_xlim_au
        else:
            xlim_max = uniform_xlim_max

        if not current_df.empty:
            sc = ax.scatter(current_df[x_col], current_df[y_col], c=colors, alpha=0.7, 
                            s=current_df['h'].apply(lambda x: 240/x if x > 0 else 60), picker=True)
            try:
                sc.set_picker(10)
            except Exception:
                pass
            point_metadata.append((ax, sc, current_df.reset_index(drop=True)))
            
            for _, row in current_df.iterrows():
                if row[x_col] < xlim_max / 4:
                    ax.annotate(
                        f"{row['des']} ({row['cd_datetime'].strftime('%Y-%m-%d')})",
                        xy=(row[x_col], row[y_col]),
                        xytext=(5, 0), textcoords='offset points',
                        ha='left', va='center', fontsize=7, color='white' 
                    )

        ax.invert_yaxis()
        
        ax.set_facecolor('#1C1C1C')
        ax.set_title(title, fontsize=12, color='white')
        ax.set_xlabel('Approach Distance (AU)', color='white')
        
        ax.set_ylabel('Abs. Mag. (h)', color='white')
        
        ax.set_xlim(left=0, right=xlim_max * 1.05)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        
        legend_handle = plt.Line2D([0], [0], marker='o', color='w', 
                                     label=label_text, markerfacecolor=colors, markersize=10)
        
        leg = ax.legend(handles=[legend_handle], loc='best')
        for text in leg.get_texts():
            text.set_color('white')
        leg.get_frame().set_facecolor('#3C3C3C')
        leg.get_frame().set_edgecolor('gray')

    # --- Graph 4: 3D Solar System Map (to asteroid belt) ---
    ax3d = axes_list[3]
    ax3d.set_facecolor('#000000')
    ax3d.set_title('Inner Solar System (Sun to Asteroid Belt)', fontsize=12, color='white')
    for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
        axis.label.set_color('white')
        axis.set_tick_params(colors='white')
    for spine in getattr(ax3d, 'spines', {}).values():
        spine.set_edgecolor('white')

    # Sun at origin
    ax3d.scatter([0], [0], [0], color='orange', s=120, marker='o')

    planets = build_planet_catalog()
    sizes = log_scale_sizes([p['radius_km'] for p in planets], min_size=30, max_size=90)
    planet_colors = {"Mercury": 'silver', "Venus": 'green', "Earth": 'blue', "Mars": 'red'}

    # Draw asteroid belt subtle band (2.2–3.2 AU) on ecliptic plane
    try:
        r_inner, r_outer = 2.2, 3.2
        theta_b = np.linspace(0, 2*np.pi, 240)
        r_b = np.linspace(r_inner, r_outer, 20)
        Theta_b, R_b = np.meshgrid(theta_b, r_b)
        X_b = R_b * np.cos(Theta_b)
        Y_b = R_b * np.sin(Theta_b)
        Z_b = np.zeros_like(X_b)
        ax3d.plot_surface(X_b, Y_b, Z_b, rstride=1, cstride=1, color='red', alpha=0.08, linewidth=0, antialiased=False)
    except Exception:
        pass

    # Draw planet orbits (bright yellow) and positions with arrows
    max_range = 2.2  # AU up to the inner edge of asteroid belt (~2.2 AU)
    theta = np.linspace(0, 2*np.pi, 800)
    # Use Skyfield for accurate current positions if possible
    sf_positions = get_planet_positions_au_with_skyfield()
    for p, s in zip(planets, sizes):
        # Orbit curve built from elements (stylized)
        Ms = np.rad2deg(theta)
        xs, ys, zs = [], [], []
        for Mdeg in Ms:
            x, y, z = orbital_elements_to_position_au(p['a'], p['e'], p['i'], p['om'], p['w'], Mdeg)
            xs.append(x); ys.append(y); zs.append(z)
        ax3d.plot(xs, ys, zs, color='yellow', lw=1.0)

        # Current position: Skyfield accurate if available, else element-based
        if sf_positions and p['name'] in sf_positions:
            x0, y0, z0 = sf_positions[p['name']]
        else:
            x0, y0, z0 = orbital_elements_to_position_au(p['a'], p['e'], p['i'], p['om'], p['w'], p['ma'])
        ax3d.scatter([x0], [y0], [z0], color=planet_colors.get(p['name'], 'white'), s=s, label=p['name'])
        # Direction arrow: small segment tangent; use next point along curve
        x1, y1, z1 = orbital_elements_to_position_au(p['a'], p['e'], p['i'], p['om'], p['w'], p['ma'] + 1.0)
        ax3d.quiver(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0, color='yellow', length=0.2, normalize=True)

    ax3d.set_xlabel('X (AU)')
    ax3d.set_ylabel('Y (AU)')
    ax3d.set_zlabel('Z (AU)')
    ax3d.set_xlim(-max_range, max_range)
    ax3d.set_ylim(-max_range, max_range)
    ax3d.set_zlim(-max_range/5, max_range/5)
    try:
        ax3d.set_box_aspect((1, 1, 0.3))
    except Exception:
        pass

    # Move planet labels outside the plot area (horizontal below)
    leg3d = ax3d.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4)
    for text in leg3d.get_texts():
        text.set_color('white')
    leg3d.get_frame().set_facecolor('#3C3C3C')
    leg3d.get_frame().set_edgecolor('gray')

    # Placeholder for selected asteroid orbit line so we can update it
    selected_orbit_line = {"line": None, "arrows": []}

    def on_pick(event):
        artist = event.artist
        for axp, scp, dfp in point_metadata:
            if artist == scp:
                ind = event.ind
                if len(ind) == 0:
                    return
                idx = int(ind[0])
                row = dfp.iloc[idx]
                
                des = str(row.get('des'))
                print(f"\n--- Pick Event ---")
                print(f"Clicked on: {des}")

                elements = try_fetch_sbdb_elements(des)
                
                if not elements:
                    print(f"ERROR: Could not fetch orbital elements for {des}. Trajectory not drawn.")
                    return 
                
                print(f"Successfully fetched elements for {des}. Drawing orbit...")
                
                # Adjust mean anomaly to current epoch if mean motion and epoch are available
                try:
                    # Use .get() for safety, though 'n' and 'epoch' might be None
                    if elements.get('n') is not None and elements.get('epoch') is not None:
                        jd_now = julian_date_utc()
                        delta_days = jd_now - float(elements['epoch'])
                        elements['ma'] = elements['ma'] + elements['n'] * delta_days
                except Exception as e:
                    print(f"Warning: Could not adjust mean anomaly: {e}")
                    pass
                # Generate orbit curve points for display
                Ms = np.linspace(0, 360, 600)
                xs, ys, zs = [], [], []
                for Mdeg in Ms:
                    x, y, z = orbital_elements_to_position_au(elements['a'], elements['e'], elements['i'], elements['om'], elements['w'], Mdeg)
                    xs.append(x); ys.append(y); zs.append(z)
                # Update or create red dotted orbit
                if selected_orbit_line["line"] is not None:
                    try:
                        selected_orbit_line["line"].remove()
                    except Exception:
                        pass
                # Remove previous arrows
                if selected_orbit_line.get("arrows"):
                    for arr in selected_orbit_line["arrows"]:
                        try:
                            arr.remove()
                        except Exception:
                            pass
                    selected_orbit_line["arrows"] = []
                line = ax3d.plot(xs, ys, zs, color='red', ls='--', lw=2.0, alpha=0.9)[0]
                selected_orbit_line["line"] = line
                # Add small direction arrows along trajectory
                try:
                    num_arrows = 10
                    idxs = np.linspace(0, len(xs) - 2, num_arrows, dtype=int)
                    arrows = []
                    for ii in idxs:
                        x0a, y0a, z0a = xs[ii], ys[ii], zs[ii]
                        x1a, y1a, z1a = xs[ii+1], ys[ii+1], zs[ii+1]
                        dx, dy, dz = (x1a - x0a), (y1a - y0a), (z1a - z0a)
                        # Normalize and scale arrow length
                        norm = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-9
                        scale = 0.25
                        q = ax3d.quiver(x0a, y0a, z0a, (dx/norm)*scale, (dy/norm)*scale, (dz/norm)*scale,
                                        color='red', alpha=0.7, length=1.0, normalize=False)
                        arrows.append(q)
                    selected_orbit_line["arrows"] = arrows
                except Exception:
                    pass
                # Expand axes to fit new orbit if needed
                try:
                    xr = max(abs(min(xs)), abs(max(xs)))
                    yr = max(abs(min(ys)), abs(max(ys)))
                    zr = max(abs(min(zs)), abs(max(zs)))
                    rng = max(max_range, xr, yr)
                    rng = min(rng * 1.1, 3.5)
                    ax3d.set_xlim(-rng, rng)
                    ax3d.set_ylim(-rng, rng)
                    ax3d.set_zlim(-max(rng/5, zr*1.2), max(rng/5, zr*1.2))
                except Exception:
                    pass
                fig.canvas.draw_idle()
                break

    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # --- Saving and Finishing ---
    print("Saving plot and cleaning up...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = OUTPUT_DIR / f"asteroid_close_approaches_{timestamp}.png"
    
    plt.tight_layout(rect=[0, 0.03, 0.96, 0.96], h_pad=5.0, w_pad=1.5)
    
    fig.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='black') 
    print(f"Saved plot to {output_file}")
    
    cleanup_old_files(OUTPUT_DIR, RETENTION_DAYS)
    
    # --- Show the plot ---
    print("Showing plot...")
    maximize_figure_window(fig)
    plt.show()

# --- 4. MAIN EXECUTION ---

# Global variable to hold the processed data
pipeline_results = None

def main():
    # --- Part 1: Start the Loading GUI (Main Thread) ---
    root, status_label, progress = create_loading_window()
    
    def on_pipeline_complete():
        root.after(100, root.destroy) # Close the loading window

    # --- Part 2: Start the Data Pipeline (Worker Thread) ---
    pipeline_thread = threading.Thread(
        target=run_data_pipeline,
        args=(status_label, progress, on_pipeline_complete),
        daemon=True
    )
    pipeline_thread.start()

    # --- Part 3: Run the Loading GUI (Main Thread) ---
    print("Showing loading window...")
    root.mainloop()
    print("Loading window closed.")

### JUPYTER MARKDOWN CELL ###
# Verify Data Pipeline Results
# Before proceeding to visualization, it's important to verify that the data pipeline successfully created and populated all four filtered dataframes. This cell checks for the existence of the global variables and displays how many asteroid records are in each category.

    # --- Part 4: Create and Show the Matplotlib Plot (Main Thread) ---
    if pipeline_results:
        try:
            create_and_show_plot(*pipeline_results)
        except Exception as e:
            print(f"Could not create or show plot: {e}")
    else:
        print("No plot was generated because pipeline failed or returned no data.")

if __name__ == "__main__":
    main()