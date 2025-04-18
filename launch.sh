#!/usr/bin/env bash

# Exits if error occurs
set -e

# Install necessary packages
apt-get update && apt-get install -y \
    x11vnc \
    xvfb \
    novnc \
    openbox \
    websockify \
    net-tools \
    nano \
    wget \
    supervisor

# Install jupyter
/isaac-sim/kit/python/bin/python3 -m pip install jupyter

# Setup virtual display
Xvfb :1 -screen 0 1920x1080x24 -ac &
export DISPLAY=:1

# Start VNC server
x11vnc -display :1 -nopw -forever -shared &

# Setup noVNC (web-based VNC client)
mkdir -p /opt/novnc
wget -qO- https://github.com/novnc/noVNC/archive/v1.2.0.tar.gz | tar xz --strip 1 -C /opt/novnc
# Start noVNC
websockify -D --web=/opt/novnc 6080 localhost:5900

# Start window manager
openbox &

echo ""
echo "==================================================="
echo "VNC server running on port 5900"
echo "Web VNC client available at http://localhost:6080/vnc.html"
echo "Jupyter running at http://localhost:8888"
echo "==================================================="
echo ""

# Start Jupyter Lab
./_isaac_sim/python.sh -m jupyter lab /workspace/isaaclab/generate_dataset.ipynb --allow-root --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.default_url='/tree/generate_dataset.ipynb'
