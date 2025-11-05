# Asteroid Hazard Assessment Dashboard

An interactive dashboard for visualizing and assessing asteroid hazards using NASA's JPL Small-Body Database API. This project provides a 3D solar system visualization where you can explore asteroid orbits and their proximity to Earth.

## Problem Motivation

On average, about one 10m asteroid passes within lunar distance of the Earth each day. Some are too small to be detected before they pass, and a still larger portion are suspected to pass by whilst escaping detection altogether. Most people are unaware of these cosmic encounters, and those that do make the news are largely sensationalized, leading to public panic, or overly technical, causing a lack of accessibility.

This project addresses these challenges by:
- Automatically fetching and processing current asteroid approach data
- Organizing temporal patterns to identify trends
- Applying scientific criteria to assess actual risk levels
- Generating clear, contextual explanations suitable for public consumption

## Features

- **Interactive 3D Solar System**: Visualize asteroid orbits in 3D space relative to the inner planets (Mercury through Mars)
- **Real-time Data**: Fetches up to 180 days of asteroid data from NASA's JPL Close Approach API in 6 batches
- **Interactive Selection**: Click on any asteroid in the 2D scatter plots to view its full 3D orbit overlaid on the solar system
- **Hazard Assessment**: Automatically identifies potentially hazardous asteroids (PHAs) using scientific criteria
- **Multiple Visualizations**: Three 2D scatter plots showing different time windows and hazard categories
- **Accurate Planet Positions**: Uses Skyfield library for high-precision planetary positions
- **Orbital Element Caching**: Caches asteroid orbital data for 30 days to reduce API load
- **Automatic Plot Saving**: Saves generated visualizations to a `plots/` directory with timestamps

## Requirements

### Required Libraries

Install the required libraries using pip:

```bash
pip install requests urllib3 pandas matplotlib numpy skyfield
```

### Library Descriptions

- **requests**: Networking with the NASA API
- **urllib3**: Creating robust, retrying connections
- **pandas**: Data analysis and manipulation
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical operations and space mathematics
- **skyfield**: High-precision astronomical calculations

### Tkinter Dependency

This project uses `tkinter` to create the graphical loading window. Tkinter is included with most standard Python installations.

**If you encounter an error like `No module named _tkinter`**, you may need to install it at the system level:

- **On Debian/Ubuntu Linux**: `sudo apt-get install python3-tk`
- **On Fedora Linux**: `sudo dnf install python3-tkinter`
- **On macOS/Windows**: Tkinter should be included by default

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd Project2-github
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install individually as shown in the Requirements section above.

## Usage

### Running the Dashboard

To launch the asteroid hazard assessment dashboard:

```bash
python "Project 2.py"
```

**Note**: The filename includes a space, so make sure to use quotes or escape the space.

### Loading Process

- A loading window will appear first with a progress bar
- The program fetches 6 batches of asteroid data (15-30 day chunks) from NASA's JPL Close Approach API
- Data is fetched for asteroids within 0.5 AU of Earth over the next 180 days
- This may take **10-30 seconds** depending on your network connection
- Once data is processed, the loading window closes and the interactive dashboard appears

### Dashboard Layout

The dashboard displays:
1. **Left Panel**: Information about the data source and PHA criteria
2. **Top-Left Plot**: Non-Hazardous, Relatively Close asteroids (within 10 lunar distances, next 60 days)
3. **Top-Right Plot**: Top 25 Hazardous asteroids (8-30 days out)
4. **Bottom-Left Plot**: Top 25 Hazardous asteroids (31-180 days out)
5. **Bottom-Right Plot**: Interactive 3D Solar System map showing inner planets and their orbits

All 2D plots show:
- **X-axis**: Approach Distance (AU)
- **Y-axis**: Absolute Magnitude (H)
- **Color coding**: Red for potentially hazardous, Gray for non-hazardous

## How to Interact

### Primary Interaction

The main feature of this dashboard is the **interactive 3D solar system**:

1. Click on any asteroid (red or gray dot) in one of the three 2D scatter plots
2. The program instantly fetches that asteroid's orbital data from NASA's Small-Body Database (cached for 30 days)
3. The full 3D orbit is drawn as a red dashed line with directional arrows on the 3D Solar System map
4. The 3D view automatically adjusts to fit the selected asteroid's orbit

This allows you to instantly visualize the trajectory of any hazardous or interesting asteroid relative to the inner planets (Mercury, Venus, Earth, Mars).

### Potentially Hazardous Asteroid (PHA) Criteria

An asteroid is classified as potentially hazardous if it meets **both** criteria:
- **H ≤ 22.0**: Absolute magnitude of 22.0 or brighter (typically ~140 meters or larger)
- **Distance ≤ 0.05 AU**: Closest approach distance of 0.05 AU (~19.5 lunar distances) or less

### Toolbar Controls

Use the standard matplotlib toolbar at the top of the window:

- **Zoom**: Use the magnifying glass icon
- **Pan**: Use the four-way arrow icon
- **Reset View**: Press the 'Home' icon
- **Save Plot**: Press the 'Save' icon to export the plot as an image
- **Exit**: Simply close the plot window

## Project Structure

```
Project2-github/
├── Project 2.py         # Main application file
├── README.md            # This file
├── plots/               # Output directory for saved visualizations (auto-created)
│   └── *.png            # Timestamped plot files (auto-cleaned after 3 days)
└── cache/               # Cache directory for orbital elements (auto-created)
    └── sbdb/            # Cached asteroid orbital data (30-day TTL)
        └── *.json       # Individual asteroid orbital element files
```

## License

This project is part of an academic course at OSU (Astro 1221).

## Contributing

This is an academic project. Contributions are welcome, but please ensure any changes align with the course requirements.

## Data Sources

- **Close Approach Data**: [NASA JPL Close Approach Data API](https://ssd-api.jpl.nasa.gov/cad.api)
- **Orbital Elements**: [NASA JPL Small-Body Database API](https://ssd-api.jpl.nasa.gov/sbdb.api)
- **Planetary Positions**: Calculated using Skyfield library with DE421 ephemeris

## Technical Details

### Data Processing
- Fetches asteroid data in 6 batches (15-30 day windows) to optimize API performance
- Processes up to 10,000 close approaches per batch
- Filters and categorizes asteroids based on time windows and hazard criteria
- Calculates derived metrics (distance in km, lunar distances, etc.)

### Visualization
- Planet orbits displayed using accurate Keplerian elements
- Current planet positions use Skyfield for high-precision calculations
- Asteroid belt region (2.2-3.2 AU) shown as a subtle band
- Interactive click handling enables dynamic orbit visualization

### Caching System
- Orbital elements cached for 30 days to reduce API calls
- Plot files automatically cleaned after 3 days
- Cache directory structure: `cache/sbdb/{asteroid_designation}.json`

## Acknowledgments

- Data provided by [NASA JPL Small-Body Database](https://ssd-api.jpl.nasa.gov/)
- Built for educational purposes in OSU's Astronomy 1221 course
- Uses [Skyfield](https://rhodesmill.org/skyfield/) for high-precision astronomical calculations

