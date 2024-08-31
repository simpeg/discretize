#!/bin/sh
set -x

sudo apt update
# Install items for headless pyvista display.
sudo apt-get install -y \
  libglx-mesa0 \
  libgl1 \
  xvfb \
  x11-xserver-utils

# qt dependents
sudo apt-get install -y \
  libdbus-1-3 \
  libegl1 \
  libopengl0 \
  libosmesa6 \
  libxcb-cursor0 \
  libxcb-icccm4 \
  libxcb-image0 \
  libxcb-keysyms1 \
  libxcb-randr0 \
  libxcb-render-util0 \
  libxcb-shape0 \
  libxcb-xfixes0 \
  libxcb-xinerama0 \
  libxcb-xinput0 \
  libxkbcommon-x11-0 \
  mesa-utils \
  x11-utils

which Xvfb
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &

# Debugging commands:
# ls -l /etc/init.d/
# sh -e /etc/init.d/xvfb start
# give xvfb some time to start
sleep 3
set +x
