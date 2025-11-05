# ===============================================
# Asteroid Hazard Assessment Dashboard
# requirements.txt
# ===============================================

# --- 1. Required Libraries ---

# To run this project, please make sure you have the required libraries installed.
# You can install them using the following command in your terminal:
#
#     pip install requests urllib3 pandas matplotlib numpy skyfield
#
# These libraries are necessary for:
#   - Networking with the NASA API (requests)
#   - Creating a robust, retrying connection (urllib3)
#   - Data analysis and manipulation (pandas)
#   - Plotting and visualization (matplotlib)
#   - Numerical operations and "space math" (numpy)
#   - High-precision astronomical calculations (skyfield)

# --- 2. How to Run ---

# After installing the libraries, you can run the project by executing the following command:
#
#     python Project2.py
#
# (Replace 'Project2.py' with the actual name of your Python file.)
#
# This will launch the asteroid hazard assessment.
#
# Please note that a loading window will appear first.
# The program must fetch and process up to 180 days of asteroid data from NASA,
# which may take 10-30 seconds depending on your network.
#
# Once the data is processed, the loading window will close, and the
# full interactive dashboard will appear.

# --- 3. How to Interact ---

# ** PRIMARY INTERACTION **
#
# The main feature of this dashboard is the interactive 3D solar system:
#
#   1. Click on any asteroid (red or gray dot) in one of the three 2D scatter plots.
#   2. The program will instantly fetch that asteroid's orbital data.
#   3. The full 3D orbit will be drawn as a red dashed line on the 3D Solar System map.
#
# This allows you to instantly visualize the trajectory of any hazardous or
# interesting asteroid relative to the inner planets.
#
# ** TOOLBAR **
#
# You can also use the standard matplotlib toolbar at the top of the window:
#   - To zoom, use the magnifying glass icon.
#   - To pan the view, use the four-way arrow icon.
#   - To reset the view, press the 'Home' icon.
#   - To save the plot as an image, press the 'Save' icon.
#   - To exit the program, simply close the plot window.

# --- 4. IMPORTANT (Tkinter Dependency) ---

# This script uses 'tkinter' to create the graphical loading window.
# 'tkinter' is included with most standard Python installations.
#
# However, if you get an error like 'No module named _tkinter',
# you may need to install it at the system level (this is common on Linux).
#
#   On Debian/Ubuntu Linux:   sudo apt-get install python3-tk
#   On Fedora Linux:          sudo dnf install python3-tkinter
