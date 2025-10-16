#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Update and install required dependencies
sudo apt update && sudo apt install -y software-properties-common

# Add deadsnakes PPA for Python 3.10
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

sudo apt install -y python3.10 python3.10-venv python3.10-distutils

# Install Python 3.11 and required libraries
sudo apt install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

# Create a virtual environment using venv
python3.10 -m venv myenv

# Ensure the virtual environment exists
if [ ! -d "myenv" ]; then
    echo "Error: Virtual environment 'myenv' was not created successfully."
    exit 1
fi

# Activate the virtual environment
source myenv/bin/activate || {
    echo "Error: Failed to activate the virtual environment. Attempting to fix..."
    curl https://bootstrap.pypa.io/get-pip.py | python
    source myenv/bin/activate
}

curl https://bootstrap.pypa.io/get-pip.py | python

# Upgrade pip in the virtual environment
pip install --upgrade pip

# Install required Python packages from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt file not found."
    exit 1
fi

echo "Environment setup complete. To activate the environment, run: source myenv/bin/activate"