#!/bin/bash

# This script installs the NVIDIA driver on Ubuntu
sudo apt install -y ubuntu-drivers-common

# Update the package list
sudo apt update

# Install the recommended NVIDIA driver
sudo ubuntu-drivers install

# Reboot the system to apply changes
echo "Rebooting the system to apply changes..."
sudo reboot