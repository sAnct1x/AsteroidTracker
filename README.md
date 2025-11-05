# Asteroid Hazard Assessment Dashboard

An interactive dashboard for visualizing and assessing asteroid hazards using NASA's Near Earth Object (NEO) API. This project provides a 3D solar system visualization where you can explore asteroid orbits and their proximity to Earth.

## Features

- **Interactive 3D Solar System**: Visualize asteroid orbits in 3D space relative to the inner planets
- **Real-time Data**: Fetches up to 180 days of asteroid data from NASA's API
- **Interactive Selection**: Click on any asteroid in the 2D scatter plots to view its full 3D orbit
- **Hazard Assessment**: Identify and visualize potentially hazardous asteroids

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
python Project2.py
```

**Note**: Replace `Project2.py` with the actual name of your Python file if it differs.

### Loading Process

- A loading window will appear first
- The program fetches and processes up to 180 days of asteroid data from NASA
- This may take **10-30 seconds** depending on your network connection
- Once data is processed, the loading window closes and the interactive dashboard appears

## How to Interact

### Primary Interaction

The main feature of this dashboard is the **interactive 3D solar system**:

1. Click on any asteroid (red or gray dot) in one of the three 2D scatter plots
2. The program instantly fetches that asteroid's orbital data
3. The full 3D orbit is drawn as a red dashed line on the 3D Solar System map

This allows you to instantly visualize the trajectory of any hazardous or interesting asteroid relative to the inner planets.

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
├── Project2.py          # Main application file
├── README.md            # This file
└── requirements.txt     # Python dependencies (if included)
```

## License

This project is part of an academic course at OSU (Astro 1221).

## Contributing

This is an academic project. Contributions are welcome, but please ensure any changes align with the course requirements.

## Acknowledgments

- Data provided by [NASA's Near Earth Object Web Service](https://api.nasa.gov/)
- Built for educational purposes in OSU's Astronomy 1221 course

