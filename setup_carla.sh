#!/usr/bin/env bash

# Detect operating system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    OS="Windows"
else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

echo "Detected operating system: $OS"

# Download and install CARLA
mkdir carla
cd carla

if [[ "$OS" == "Linux" ]]; then
    echo "Downloading CARLA for Linux..."
    wget -O carla.tar.gz hhttps://tiny.carla.org/carla-0-10-0-linux-tar
    tar -xf carla.tar.gz
    rm carla.tar.gz
elif [[ "$OS" == "Windows" ]]; then
    echo "Downloading CARLA for Windows..."
    # Replace with actual Windows download URLs
    wget -O carla.zip https://tiny.carla.org/carla-0-10-0-windows
    # Use appropriate unzip command for Windows
    unzip carla.zip
    rm carla.zip
fi

cd ..